"""Print a benchmark JSONL (pp_bench.py output) as a table: one row per (prompt, tokens, conc) x target.

  perf_table.py file.jsonl [--md] [--filter 'conc=1,tokens=32']
"""
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
md = "--md" in sys.argv
flt = {}
for a in sys.argv[1:]:
    if a.startswith("--filter"):
        pass
if "--filter" in sys.argv:
    for kv in sys.argv[sys.argv.index("--filter") + 1].split(","):
        k, v = kv.split("=")
        flt[k] = v
rows = [r for r in (json.loads(l) for l in open(args[0]) if l.startswith("{")) if "ttft_ms" in r]
rows = [r for r in rows if all(str(r[k]) == v for k, v in flt.items())]
if md:
    print("| prompt | tokens | C | target | TTFT ms (p25 / **median** / p75) | TPOT ms (p25 / **median** / p75) | out tok/s |")
    print("|---|---|---|---|---|---|---|")
else:
    print(f"{'prompt':>6} {'tok':>3} {'C':>2} {'target':>12} {'TTFT p25/med/p75 ms':>24} {'TPOT p25/med/p75 ms':>24} {'lat med ms':>10} {'out tok/s':>9}")
for r in rows:
    t, p = r["ttft_ms"], r["tpot_ms"]
    if md:
        print(f"| {r['prompt']} | {r['tokens']} | {r['conc']} | {r['target']} | {t['p25']:.1f} / **{t['median']:.1f}** / {t['p75']:.1f} | "
              f"{p['p25']:.2f} / **{p['median']:.2f}** / {p['p75']:.2f} | {r['out_tok_s']['median']:.1f} |")
    else:
        print(f"{r['prompt']:>6} {r['tokens']:>3} {r['conc']:>2} {r['target']:>12} {t['p25']:>7.1f}/{t['median']:>6.1f}/{t['p75']:>6.1f} {p['p25']:>7.2f}/{p['median']:>6.2f}/{p['p75']:>6.2f} "
              f"{r['latency_ms']['median']:>10.1f} {r['out_tok_s']['median']:>9.1f}")
