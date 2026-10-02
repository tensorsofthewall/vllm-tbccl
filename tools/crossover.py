"""Interleaved PP matrix (perf_pp.jsonl): per (prompt, conc), CUDA->CPU vs CUDA->Metal vs Metal->CUDA, 32-token requests. Winner by end-to-end latency."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
rows = [r for r in rows if "ttft_ms" in r and r["tokens"] == 32]
cell = {}
for r in rows:
    cell.setdefault((r["prompt"], r["conc"]), {})[r["target"]] = r
print("| prompt | C | metric | CUDA->CPU | CUDA->Metal | Metal->CUDA | CUDA->Metal vs CUDA->CPU |")
print("|---|---|---|---|---|---|---|")
for (pr, c), t in sorted(cell.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
    cpu, met, mc = t["cuda_cpu"], t["cuda_metal"], t["metal_cuda"]
    for name, key in (("TTFT ms", "ttft_ms"), ("TPOT ms", "tpot_ms"), ("latency ms", "latency_ms")):
        a, b, d = cpu[key]["median"], met[key]["median"], mc[key]["median"]
        print(f"| {pr} | {c} | {name} | {a:.1f} | {b:.1f} | {d:.1f} | {'Metal ' + format(a / b, '.2f') + 'x faster' if b < a else 'CPU ' + format(b / a, '.2f') + 'x faster'} |")
