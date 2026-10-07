"""The CUDA+Metal vLLM 0.30 work characterization client: TTFT versus prompt length (max_tokens=1) and decode ms/token, against one running server.

    python tools/p70_perf.py --api http://HOST:PORT --model NAME [--lens 16,128,512,900] [--reps 5] [--decode 64]

Prompts are repeated words (no prefix-cache hits: each rep adds a unique leading number). Reports the median / p10 / p90 of non-streamed wall time per request.
"""
import argparse
import json
import statistics as st
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--api", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--lens", default="16,128,512,900")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--decode", type=int, default=64)
a = ap.parse_args()


def call(prompt, n):
    body = json.dumps({"model": a.model, "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}).encode()
    t = time.monotonic()
    d = json.load(urllib.request.urlopen(urllib.request.Request(a.api + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    return time.monotonic() - t, d["usage"]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


out = {"prefill": {}, "decode": {}}
call("warm up", 4)
for L in (int(x) for x in a.lens.split(",")):
    ts, pt = [], None
    for r in range(a.reps + 1):
        t, u = call(f"{r * 7919 + L} " + "hello " * (L - 2), 1)
        pt = u["prompt_tokens"]
        if r:
            ts.append(t)
    out["prefill"][L] = {"prompt_tokens": pt, "median_s": st.median(ts), "p10_s": q(ts, 0.1), "p90_s": q(ts, 0.9)}
ts = []
for r in range(a.reps + 1):
    t, u = call(f"{r} The history of numbers is", a.decode)
    if r:
        ts.append(t / u["completion_tokens"])
out["decode"] = {"tokens": a.decode, "ms_per_token_median": 1e3 * st.median(ts), "ms_p10": 1e3 * q(ts, 0.1), "ms_p90": 1e3 * q(ts, 0.9)}
print(json.dumps(out))
