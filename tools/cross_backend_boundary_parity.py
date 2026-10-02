"""Boundary parity through teacher-forced logits.

collect: POST each reference sequence (prompt ids + generated ids) with ``prompt_logprobs=K, max_tokens=1`` and store, per position,
         the top-K token ids/logprobs and the logprob of the actual next token.
compare: for two collected files report, per sequence, the max |logprob difference| of the actual-next-token and over the top-K
         entries both runs share, and the fraction of positions where the argmax agrees.

Used on a heterogeneous (CUDA stage -> Metal stage) engine versus single-stage CUDA and single-stage Metal engines. It validates the
stage-boundary mapping (hidden_states+residual <-> raw residual stream, row order) against the references' own cross-backend spread.
"""
import argparse
import json
import sys
import urllib.request


def collect(a):
    ref = json.load(open(a.ref))
    out = []
    for r in ref:
        ids = r["prompt_token_ids"] + r["token_ids"][: a.forced]
        body = json.dumps({"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": a.k}).encode()
        d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=300))
        pl = d["choices"][0]["prompt_logprobs"]
        rows = []
        for pos, entry in enumerate(pl):
            if entry is None:
                rows.append(None)
                continue
            rows.append({tok: v["logprob"] for tok, v in entry.items()})
        out.append({"ids": ids, "rows": rows})
    json.dump(out, open(a.out, "w"))
    print(f"collected {len(out)} sequences -> {a.out}")


def compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    worst = 0.0
    for i, (x, y) in enumerate(zip(A, B)):
        assert x["ids"] == y["ids"]
        mx_err, n, arg_ok = 0.0, 0, 0
        for pos, (ra, rb) in enumerate(zip(x["rows"], y["rows"])):
            if ra is None or rb is None:
                continue
            tok = str(x["ids"][pos])
            if tok in ra and tok in rb:
                mx_err = max(mx_err, abs(ra[tok] - rb[tok]))
            common = set(ra) & set(rb)
            for t in common:
                mx_err = max(mx_err, abs(ra[t] - rb[t]))
            n += 1
            arg_ok += max(ra, key=ra.get) == max(rb, key=rb.get)
        print(f"seq {i}: positions={n} argmax_agree={arg_ok}/{n} max|dlogprob|={mx_err:.4f}")
        worst = max(worst, mx_err)
    print(f"worst max|dlogprob| = {worst:.4f}")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("collect"); c.add_argument("--url", required=True); c.add_argument("--model", required=True)
c.add_argument("--ref", required=True); c.add_argument("--out", required=True); c.add_argument("--k", type=int, default=5)
c.add_argument("--forced", type=int, default=16); c.set_defaults(f=collect)
d = sub.add_parser("compare"); d.add_argument("a"); d.add_argument("b"); d.set_defaults(f=compare)
a = p.parse_args()
a.f(a)
