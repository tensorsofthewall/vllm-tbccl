"""Phased client for a running vLLM PP=2 deployment (correctness against the single-rank reference, concurrency, cancellation, timing).

    python tools/p69_client.py --api http://HOST:PORT --model PATH --phase short|medium|sequential|concurrent|cancel --ref docs/data/phase69/ref_linux_cuda.json --out FILE.json

Strict controls are the prompts whose argmax margin is comfortably non-zero on both devices (numbers, medium, python); every phase compares generated token strings to the
single-rank reference token for token. Timing: streamed TTFT / inter-token times (one streamed request per phase), request latency and aggregate throughput under concurrency.
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics as st
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p69_gen  # noqa: E402

STRICT = ("numbers", "medium", "python")


def stream(api, model, prompt, n, drop_after=None):
    req = urllib.request.Request(api + "/v1/completions", method="POST", headers={"content-type": "application/json"},
                                 data=json.dumps({"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0, "stream": True}).encode())
    t0 = time.monotonic()
    times, text = [], ""
    r = urllib.request.urlopen(req, timeout=600)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        ch = json.loads(line[5:])["choices"][0]
        if ch.get("text"):
            times.append(time.monotonic() - t0)
            text += ch["text"]
        if drop_after and len(times) >= drop_after:
            break
    r.close()
    return times, text


def check(api, model, ref, names, n=16):
    out = {}
    for name in names:
        t0 = time.monotonic()
        d = p69_gen.call(api, model, p69_gen.PROMPTS[name], n)
        lat = time.monotonic() - t0
        toks = d["choices"][0]["logprobs"]["tokens"]
        out[name] = {"match": toks == ref[name]["tokens"], "latency_s": lat, "completion_tokens": d["usage"]["completion_tokens"], "prompt_tokens": d["usage"]["prompt_tokens"], "text": d["choices"][0]["text"][:80]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    res = {"phase": a.phase}
    if a.phase == "short":
        res["checks"] = check(a.api, a.model, ref, ["numbers"])
    elif a.phase == "medium":
        res["checks"] = check(a.api, a.model, ref, ["medium", "python"])
        times, _ = stream(a.api, a.model, p69_gen.PROMPTS["medium"], 16)
        gaps = [b - c for c, b in zip(times, times[1:])]
        res["stream_medium"] = {"ttft_s": times[0], "tpot_median_s": st.median(gaps), "tpot_p95_s": sorted(gaps)[int(0.95 * (len(gaps) - 1))], "tokens": len(times)}
    elif a.phase == "sequential":
        res["checks"] = {f"{n}#{i}": v for i in range(2) for n, v in check(a.api, a.model, ref, STRICT).items()}
    elif a.phase == "concurrent":
        jobs = [STRICT[i % 3] for i in range(12)]
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(12) as ex:
            outs = list(ex.map(lambda n: (n, p69_gen.call(a.api, a.model, p69_gen.PROMPTS[n], 16)), jobs))
        wall = time.monotonic() - t0
        res["concurrent"] = {"requests": 12, "all_match": all(d["choices"][0]["logprobs"]["tokens"] == ref[n]["tokens"] for n, d in outs), "wall_s": wall,
                             "generated_tokens": sum(d["usage"]["completion_tokens"] for _, d in outs), "tokens_per_s": sum(d["usage"]["completion_tokens"] for _, d in outs) / wall}
    elif a.phase == "cancel":
        times, _ = stream(a.api, a.model, p69_gen.PROMPTS["count"], 400, drop_after=4)
        time.sleep(3)
        res["cancel"] = {"chunks_before_drop": len(times)}
        res["checks"] = check(a.api, a.model, ref, ["numbers"])
    else:
        raise SystemExit("unknown phase")
    ok = all(v["match"] for v in res.get("checks", {}).values()) and res.get("concurrent", {}).get("all_match", True)
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "checks"} | {"checks": {k: (v["match"], round(v["latency_s"], 2)) for k, v in res.get("checks", {}).items()}}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
