"""Single-rank vLLM reference run: FP32, eager, greedy. Writes token ids/text to --out (JSON)."""
import argparse
import json
import time

from vllm import LLM, SamplingParams

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--mem", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--lp-out", default=None, help="after generation, teacher-force prompt+generated and store top-K prompt logprobs")
    p.add_argument("--lp-k", type=int, default=20)
    a = p.parse_args()

    PROMPTS = ["The capital of France is", "In a distant future, robots and humans lived together and",
               "def fibonacci(n):\n    "]
    mem = a.mem if a.mem is not None else (0.2 if a.device in ("cpu", "metal") else 0.5)
    t0 = time.monotonic()
    llm = LLM(model=a.model, dtype=a.dtype, enforce_eager=True, max_model_len=a.max_model_len, seed=0, gpu_memory_utilization=mem)
    t1 = time.monotonic()
    if a.prompts_file:
        PROMPTS = [o.get("prompt_token_ids") and {"prompt_token_ids": o["prompt_token_ids"]} or o["prompt"] for o in json.load(open(a.prompts_file))]
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=a.max_tokens))
    print(f"startup_s={t1 - t0:.1f} generate_s={time.monotonic() - t1:.2f}")
    res = [{"prompt": o.prompt, "prompt_token_ids": list(o.prompt_token_ids), "token_ids": list(o.outputs[0].token_ids),
            "text": o.outputs[0].text} for o in outs]
    for r in res:
        print(json.dumps(r))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if a.lp_out:
        seqs = [r["prompt_token_ids"] + r["token_ids"] for r in res]
        lps = llm.generate([{"prompt_token_ids": q} for q in seqs], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=a.lp_k))
        rows = [[None if e is None else {str(t): v.logprob for t, v in e.items()} for e in o.prompt_logprobs] for o in lps]
        json.dump([{"ids": q, "rows": r} for q, r in zip(seqs, rows)], open(a.lp_out, "w"))


if __name__ == "__main__":
    main()
