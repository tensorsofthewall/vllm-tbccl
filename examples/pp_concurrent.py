"""Concurrent (optionally staggered) greedy requests against a running server, checked against a reference.

Every request uses the reference's prompt token ids. A request whose tokens differ from the reference is classified by the reference's
own teacher-forced distribution at the first divergent position: top-1/top-2 logprob margin <= --tie (default 0.25) is a bf16 near-tie
(batch composition changes bf16 numerics), anything larger is UNEXPLAINED and fails the run. Also reports wall time and throughput.
"""
import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--model", required=True)
p.add_argument("--ref", required=True, help="<tag>_ref.json; <tag>_lp.json next to it supplies the margins")
p.add_argument("--conc", type=int, default=4)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--tokens", type=int, default=32)
p.add_argument("--stagger", type=float, default=0.0, help="random start delay in [0, stagger] seconds")
p.add_argument("--random-len", action="store_true", help="vary max_tokens per request (3..tokens) and compare the shared prefix")
p.add_argument("--tie", type=float, default=0.25)
p.add_argument("--only", default=None, help="comma-separated reference indices to use")
a = p.parse_args()
ref = json.load(open(a.ref))
lp = json.load(open(a.ref.replace("_ref.json", "_lp.json")))
idx = [int(i) for i in a.only.split(",")] if a.only else list(range(len(ref)))
jobs = [i for _ in range(a.repeat) for i in idx]
jobs = (jobs * ((a.conc + len(jobs) - 1) // len(jobs)))[: max(a.conc, len(jobs))] if a.repeat == 1 and len(jobs) < a.conc else jobs
rng = random.Random(7)
plan = [(i, rng.randint(3, a.tokens) if a.random_len else a.tokens, rng.random() * a.stagger) for i in jobs]


def margin(row):
    v = sorted(row.values(), reverse=True)
    return v[0] - v[1]


def go(job):
    i, n, delay = job
    time.sleep(delay)
    body = json.dumps({"model": a.model, "prompt": ref[i]["prompt_token_ids"], "max_tokens": n, "temperature": 0, "ignore_eos": True,
                       "return_token_ids": True}).encode()
    t = time.monotonic()
    d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=600))
    return i, n, d["choices"][0]["token_ids"], time.monotonic() - t


t0 = time.monotonic()
with cf.ThreadPoolExecutor(len(plan)) as ex:
    out = list(ex.map(go, plan))
wall = time.monotonic() - t0
exact = tie = bad = 0
for i, n, toks, _ in out:
    want = ref[i]["token_ids"][:n]
    if toks == want:
        exact += 1
        continue
    d = next(j for j, (u, v) in enumerate(zip(toks, want)) if u != v) if len(toks) == len(want) else min(len(toks), len(want))
    P = len(ref[i]["prompt_token_ids"])
    m = margin(lp[i]["rows"][P + d])
    if m <= a.tie:
        tie += 1
    else:
        bad += 1
        print(f"UNEXPLAINED: ref {i} diverges at gen token {d}, reference margin {m:.3f}")
total = sum(len(t) for _, _, t, _ in out)
lat = sorted(l for *_, l in out)
print(f"requests={len(out)} exact={exact} near_tie={tie} unexplained={bad} wall={wall:.2f}s out_tok/s={total / wall:.1f} "
      f"latency median={lat[len(lat) // 2]:.2f}s max={lat[-1]:.2f}s")
raise SystemExit(1 if bad else 0)
