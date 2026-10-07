"""Markdown performance tables for, built from docs/data/phase48/perf_single.jsonl and perf_pp.jsonl.

Single-stage rows come from one interleaved session (cuda/metal/cpu), PP rows from another (cuda_cpu/cuda_metal/metal_cuda); inside a session the
targets are interleaved round by round, between sessions they are not (state this when quoting a cross-session ratio).
"""
import json
import sys

D = "docs/data/phase48/"
NAMES = [("cuda", "CUDA single-stage", "single"), ("metal", "Metal single-stage", "single"), ("cpu", "CPU single-stage", "single"),
         ("cuda_cpu", "CUDA -> CPU PP (14/14)", "pp"), ("cuda_metal", "CUDA -> Metal PP (14/14)", "pp"), ("metal_cuda", "Metal -> CUDA PP (14/14)", "pp")]
rows = {}
for f, kind in (("perf_single.jsonl", "single"), ("perf_pp.jsonl", "pp")):
    for l in open(D + f):
        if l.startswith("{"):
            r = json.loads(l)
            if "ttft_ms" in r:
                rows[(kind, r["target"], r["prompt"], r["tokens"], r["conc"])] = r


ANOMALY = {("pp", "metal_cuda", "p16", 32, 1)}   # bimodal in that session (15.2 ms in round 1, ~21.8 ms after); standalone runs give ~15.6 ms


def table(prompt, conc, tokens=32):
    out = [f"| Configuration | TTFT ms | TPOT ms (= ms/token) | TPOT p25-p75 | request latency ms | output tok/s |", "|---|---|---|---|---|---|"]
    for key, label, kind in NAMES:
        r = rows[(kind, key, prompt, tokens, conc)]
        mark = " †" if (kind, key, prompt, tokens, conc) in ANOMALY else ""
        out.append(f"| {label}{mark} | {r['ttft_ms']['median']:.1f} | {r['tpot_ms']['median']:.2f} | {r['tpot_ms']['p25']:.2f}-{r['tpot_ms']['p75']:.2f} | "
                   f"{r['latency_ms']['median']:.0f} | {r['out_tok_s']['median']:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for prompt in ("p16", "p128", "p512"):
        print(f"\n#### {prompt} ({rows[('single', 'cuda', prompt, 32, 1)]['prompt_tokens']}-token prompt), 32 decode tokens, concurrency 1\n")
        print(table(prompt, 1))
    for conc in (4, 12):
        for prompt in ("p16", "p128", "p512"):
            print(f"\n#### {prompt}, 32 decode tokens, concurrency {conc} ({conc} simultaneous requests per round; TTFT/TPOT are per request)\n")
            print(table(prompt, conc))
    print("\n#### 4-token requests, concurrency 1 (exposes TTFT)\n")
    for prompt in ("p16", "p128", "p512"):
        print(f"\n{prompt}:\n")
        print(table(prompt, 1, tokens=4))
