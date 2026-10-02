"""Streaming latency/throughput benchmark with interleaved rounds across targets (A/B/A/B).

  pp_bench.py --target name=url=model [--target ...] --prompt p128 --tokens 32 --conc 4 --rounds 7 --warmup 2

A round fires --conc simultaneous requests at ONE target; rounds rotate over the targets so every target sees the same drift. Servers run
with prefix caching disabled (--no-enable-prefix-caching), so every request is a full prefill of the fixed prompt; --unique-first-token is
only for servers where caching could not be disabled. Per request: TTFT (first streamed token), TPOT ((last-first)/(n-1)), latency. Per round: wall and output tok/s. ``--warmup``
rounds per target are discarded. Prints one JSON summary per target and appends raw per-request rows to --raw.
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import random
import statistics as st
import time
import urllib.parse

p = argparse.ArgumentParser()
p.add_argument("--target", action="append", required=True, help="name=url=model")
p.add_argument("--prompts", default="tests/fixtures/prompts.json")
p.add_argument("--prompt", required=True, help="prompt name in the prompts file (short0..2, p16, p128, p512)")
p.add_argument("--tokens", type=int, default=32)
p.add_argument("--conc", type=int, default=1)
p.add_argument("--rounds", type=int, default=7)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--raw", default=None)
p.add_argument("--unique-first-token", action="store_true")
a = p.parse_args()
base = next(o for o in json.load(open(a.prompts)) if o["name"] == a.prompt)
base_ids = base.get("prompt_token_ids")
if base_ids is None:
    raise SystemExit("--prompt must be a token-id prompt (p16/p128/p512)")
rng = random.Random(11)
targets = [t.split("=", 2) for t in a.target]


def one(url, model):
    ids = ([rng.randrange(1000, 100000)] + base_ids[1:]) if a.unique_first_token else base_ids
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True, "stream": True})
    t0 = time.monotonic()
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    stamps = []
    for line in r:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            stamps.append(time.monotonic())
    c.close()
    n = len(stamps)
    return {"ttft": stamps[0] - t0, "tpot": (stamps[-1] - stamps[0]) / max(n - 1, 1), "latency": stamps[-1] - t0, "chunks": n}


rows = {n: [] for n, _, _ in targets}
walls = {n: [] for n, _, _ in targets}
for rnd in range(a.rounds + a.warmup):
    for name, url, model in targets:
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            res = list(ex.map(lambda _: one(url, model), range(a.conc)))
        wall = time.monotonic() - t0
        if rnd >= a.warmup:
            rows[name] += res
            walls[name].append(wall)


def q(xs, f):
    xs = sorted(xs)
    return xs[min(int(f * len(xs)), len(xs) - 1)]


for name, _, _ in targets:
    r = rows[name]
    out = {"target": name, "prompt": a.prompt, "prompt_tokens": len(base_ids), "tokens": a.tokens, "conc": a.conc, "rounds": a.rounds, "n_req": len(r)}
    for k in ("ttft", "tpot", "latency"):
        v = [x[k] * 1e3 for x in r]
        out[k + "_ms"] = {"median": round(st.median(v), 2), "p25": round(q(v, .25), 2), "p75": round(q(v, .75), 2), "min": round(min(v), 2), "max": round(max(v), 2)}
    tps = [a.conc * a.tokens / w for w in walls[name]]
    out["out_tok_s"] = {"median": round(st.median(tps), 1), "min": round(min(tps), 1), "max": round(max(tps), 1)}
    out["chunks_ok"] = all(x["chunks"] == a.tokens for x in r)
    print(json.dumps(out))
    if a.raw:
        with open(a.raw, "a") as fh:
            for x in r:
                fh.write(json.dumps({"target": name, "prompt": a.prompt, "conc": a.conc, "tokens": a.tokens, **x}) + "\n")
