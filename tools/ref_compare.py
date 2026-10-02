"""Compare single-stage (or pipeline) greedy references and teacher-forced logprobs from two runs.

For each prompt: first token divergence, and at that position the top-1/top-2 logprob margin each run assigns on the shared prefix
(a flip with a small margin in both is a bf16 near-tie, not corruption). Also: argmax agreement over teacher-forced positions of
the SHARED token sequence, and the largest top-K logprob difference.
"""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("a")
p.add_argument("b")
p.add_argument("--names", default=None)
a = p.parse_args()
names = [o["name"] for o in json.load(open(a.names))] if a.names else None


def margin(row):
    v = sorted(row.values(), reverse=True)
    return v[0] - v[1]


def run(tag):
    return json.load(open(f"{tag}_ref.json")), json.load(open(f"{tag}_lp.json"))


(ra, la), (rb, lb) = run(a.a), run(a.b)
STATS = []
for i, (x, y) in enumerate(zip(ra, rb)):
    t1, t2, P = x["token_ids"], y["token_ids"], len(x["prompt_token_ids"])
    d = next((j for j, (u, v) in enumerate(zip(t1, t2)) if u != v), None)
    nm = names[i] if names else i
    line = f"{nm}: prompt={P}tok " + ("identical 32/32" if d is None else f"first divergence at gen token {d}")
    if d is not None:
        ra_row, rb_row = la[i]["rows"][P + d], lb[i]["rows"][P + d]
        line += f" | A top1-top2 margin {margin(ra_row):.4f}, B margin {margin(rb_row):.4f}"
    # teacher-forced agreement on the shared sequence prefix up to divergence (identical prefixes, identical positions)
    n = P + (d if d is not None else len(t1))
    agree, mx, cnt = 0, 0.0, 0
    top5 = []
    for pos in range(1, n):
        qa, qb = la[i]["rows"][pos], lb[i]["rows"][pos]
        if qa is None or qb is None:
            continue
        cnt += 1
        agree += max(qa, key=qa.get) == max(qb, key=qb.get)
        for t in set(qa) & set(qb):
            mx = max(mx, abs(qa[t] - qb[t]))
        for t in sorted(qa, key=qa.get, reverse=True)[:5]:
            if t in qb:
                top5.append(abs(qa[t] - qb[t]))
    top5.sort()
    STATS.extend(top5)
    print(line + f" | shared-prefix positions {cnt}, argmax agree {agree}/{cnt}, max|dlogprob| {mx:.4f}")
STATS.sort()
if STATS:
    print(f"ALL PROMPTS top-5 |dlogprob|: n={len(STATS)} median={STATS[len(STATS) // 2]:.4f} p90={STATS[int(.9 * len(STATS))]:.4f} "
          f"p99={STATS[int(.99 * len(STATS))]:.4f} max={STATS[-1]:.4f}")
