"""Offline pipeline-boundary parity without a network: real upstream-vLLM CUDA layers on one side, mlx-lm layers on the other.

The model is cut after layer SPLIT-1 (the vLLM PP boundary). Four paths are run per case on the SAME token sequence and compared:

  cuda-full   vLLM (CUDA) single engine; hooks record the boundary (hidden_states, residual) after layer SPLIT-1 and the logits per step
  cuda->metal boundary x = hidden_states + residual (the codec) fed to mlx-lm layers SPLIT.. + norm + head
  metal-full  mlx-lm full model teacher-forced on the vLLM-generated tokens; records its boundary x and logits per step
  metal->cuda boundary (hidden_states = x, residual = 0) injected into vLLM layer SPLIT through a forward_pre_hook

A step is one model forward: step 0 is the prefill of all prompt rows, later steps are single-row decodes with a KV cache, so a case
exercises prefill and decode. ``--wrong`` runs the negative control (residual dropped) that the comparison must flag.

Usage (the two machines exchange .safetensors/.json files; no network is involved):
  cuda-full / cuda-second  run on the CUDA host (vllm-tbccl venv)       metal-full / metal-second  run on the Mac (MLX venv)
  compare                  runs anywhere with torch
"""
import argparse
import json
import os
import sys

SPLIT = 14
STEPS = 8


def short_ids(model, cases_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    res = []
    for c in json.load(open(cases_path)):
        if c["name"] in ("short0", "short1", "p128"):
            ids = c.get("prompt_token_ids") or tok.encode(c["prompt"], add_special_tokens=False)
            res.append({"name": c["name"], "ids": ids})
    res.insert(0, {"name": "tok1", "ids": [res[0]["ids"][0]]})
    return res


# ------------------------------------------------------------------------------------------------------ CUDA side
def cuda_run(a, inject=None):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    import torch
    from safetensors.torch import load_file, save_file
    from vllm import LLM, SamplingParams

    cases = short_ids(a.model, a.prompts)
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True, max_model_len=1024, gpu_memory_utilization=a.mem, seed=0)
    rec = {"bnd": [], "logits": []}
    boundary_in = load_file(a.boundary) if inject else None
    state = {"case": None, "step": 0}

    def install(model):
        inner = model.model
        layers = inner.layers

        def out_hook(_m, _args, out):
            h, r = out
            rec["bnd"].append((h.detach().clone().cpu(), r.detach().clone().cpu()))

        layers[SPLIT - 1].register_forward_hook(out_hook)
        if inject:
            def pre_hook(_m, args):
                pos, h, r = args
                x = boundary_in[f"{state['case']}/{state['step']}/x"].to(h.device)
                assert x.shape == h.shape, (x.shape, h.shape)
                state["step"] += 1
                zeros = torch.zeros_like(x)
                return pos, x, zeros
            layers[SPLIT].register_forward_pre_hook(pre_hook)

        def logit_hook(_m, _args, out):
            rec["logits"].append(out.detach().float().cpu())
        model.logits_processor.register_forward_hook(logit_hook)
        return True

    llm.apply_model(install)
    tensors, meta = {}, {}
    for c in cases:
        rec["bnd"].clear()
        rec["logits"].clear()
        state["case"], state["step"] = c["name"], 0
        out = llm.generate([{"prompt_token_ids": c["ids"]}], SamplingParams(temperature=0, max_tokens=STEPS, ignore_eos=True))
        toks = list(out[0].outputs[0].token_ids)
        assert len(rec["bnd"]) == STEPS and len(rec["logits"]) == STEPS, (len(rec["bnd"]), len(rec["logits"]))
        meta[c["name"]] = {"ids": c["ids"], "gen": toks}
        for i, ((h, r), lg) in enumerate(zip(rec["bnd"], rec["logits"])):
            tensors[f"{c['name']}/{i}/hidden"] = h.contiguous()
            tensors[f"{c['name']}/{i}/residual"] = r.contiguous()
            tensors[f"{c['name']}/{i}/logits"] = lg.reshape(-1).contiguous()
        print(c["name"], "prompt", len(c["ids"]), "gen", toks)
    save_file(tensors, a.out)
    json.dump(meta, open(a.out + ".json", "w"))


# ------------------------------------------------------------------------------------------------------ Metal side
def metal_load(model_path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm.utils import load

    model, _ = load(model_path)
    return model


def metal_stage(model, x, caches, lo, hi, head):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    mask = create_attention_mask(x, caches[lo])
    h = x
    for i in range(lo, hi):
        h = inner.layers[i](h, mask, caches[i])
    if not head:
        return h
    h = inner.norm(h)
    out = inner.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
    return out[:, -1, :].astype(mx.float32)


def metal_run(a):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model = metal_load(a.model)
    n_layers = len(model.model.layers)
    meta = json.load(open(a.ref + ".json"))
    src = mx.load(a.ref)
    out = {}
    for name, m in meta.items():
        ids = m["ids"]
        seq = ids + m["gen"]                       # teacher-forced on the tokens the CUDA engine generated
        cache_full = make_prompt_cache(model)
        cache_second = make_prompt_cache(model)
        for step in range(STEPS):
            rows = ids if step == 0 else [seq[len(ids) + step - 1]]
            # metal-full (also records its own boundary)
            x0 = model.model.embed_tokens(mx.array([rows]))
            bnd = metal_stage(model, x0, cache_full, 0, SPLIT, head=False)
            logits_full = metal_stage(model, bnd, cache_full, SPLIT, n_layers, head=True)
            out[f"{name}/{step}/x"] = bnd[0].astype(mx.bfloat16)
            out[f"{name}/{step}/logits_full"] = logits_full[0]
            # cuda -> metal: x = hidden_states + residual from the CUDA first half
            h = src[f"{name}/{step}/hidden"]
            r = src[f"{name}/{step}/residual"]
            x = h if a.wrong else h + r
            logits_cm = metal_stage(model, x[None], cache_second, SPLIT, n_layers, head=True)
            out[f"{name}/{step}/logits_cm"] = logits_cm[0]
            mx.eval(out[f"{name}/{step}/x"], out[f"{name}/{step}/logits_full"], out[f"{name}/{step}/logits_cm"])
    mx.save_safetensors(a.out, out)
    print("saved", a.out, len(out), "tensors")


# ------------------------------------------------------------------------------------------------------ comparison
def compare(a):
    import torch
    from safetensors.torch import load_file

    cuda = load_file(a.cuda)                 # cuda-full (+ boundary)
    metal = load_file(a.metal)               # metal-full, cuda->metal
    mc = load_file(a.mc) if a.mc else None   # metal->cuda logits (cuda_run with --inject)
    meta = json.load(open(a.cuda + ".json"))

    def lsm(x):
        return torch.log_softmax(x.float(), dim=-1)

    def cmp(la, lb, k=20):
        pa, pb = lsm(la), lsm(lb)
        top = torch.topk(pa, k).indices
        return (pa.argmax() == pb.argmax()).item(), (pa[top] - pb[top]).abs().max().item()

    rows = []
    for name in meta:
        for step in range(STEPS):
            lc = cuda[f"{name}/{step}/logits"]
            lm = metal[f"{name}/{step}/logits_full"]
            lcm = metal[f"{name}/{step}/logits_cm"]
            rec = {"case": name, "step": step, "n_rows": len(meta[name]["ids"]) if step == 0 else 1}
            rec["base_c_vs_m"] = cmp(lc, lm)
            rec["cm_vs_c"] = cmp(lcm, lc)
            rec["cm_vs_m"] = cmp(lcm, lm)
            if mc is not None:
                lmc = mc[f"{name}/{step}/logits"]
                rec["mc_vs_c"] = cmp(lmc, lc)
                rec["mc_vs_m"] = cmp(lmc, lm)
            rows.append(rec)
    keys = [k for k in rows[0] if k not in ("case", "step", "n_rows")]
    print(f"{'case':7} {'step':>4} " + " ".join(f"{k:>16}" for k in keys))
    for r in rows:
        print(f"{r['case']:7} {r['step']:>4} " + " ".join(f"{('ok' if r[k][0] else 'ARGMAX!')+' %.3f' % r[k][1]:>16}" for k in keys))
    print("--- summary (argmax agreement / worst max|dlogprob| over reference top-20)")
    for k in keys:
        agree = sum(r[k][0] for r in rows)
        print(f"{k:12} argmax {agree}/{len(rows)}  worst {max(r[k][1] for r in rows):.3f}  median {sorted(r[k][1] for r in rows)[len(rows)//2]:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cuda-full", "cuda-second", "metal", "compare"])
    p.add_argument("--model")
    p.add_argument("--prompts", default="tests/fixtures/prompts.json")
    p.add_argument("--mem", type=float, default=0.5)
    p.add_argument("--out")
    p.add_argument("--ref", help="metal: the cuda-full safetensors (its .json holds the token sequences)")
    p.add_argument("--boundary", help="cuda-second: the Metal 'x' tensors (metal output)")
    p.add_argument("--wrong", action="store_true", help="negative control: drop the residual term")
    p.add_argument("--cuda")
    p.add_argument("--metal")
    p.add_argument("--mc")
    a = p.parse_args()
    if a.mode == "cuda-full":
        cuda_run(a)
    elif a.mode == "cuda-second":
        cuda_run(a, inject=True)
    elif a.mode == "metal":
        metal_run(a)
    else:
        compare(a)


if __name__ == "__main__":
    sys.exit(main())
