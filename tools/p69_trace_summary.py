"""Phase 69: what travelled through ProcessGroupTBCCL, per session and rank (from VLLM_TBCCL_TRACE files): operations by kind, device group (cuda/cpu tensors) vs control (cpu tensors), bytes."""
import collections
import glob
import json
import sys

for tag in sys.argv[1:]:
    print(f"== {tag}")
    for f in sorted(glob.glob(f"docs/data/phase69/raw/{tag}/{tag}_trace.pg.*.json")):
        ev = json.load(open(f))
        if not ev:
            continue
        by = collections.defaultdict(lambda: [0, 0])
        for e in ev:
            k = (e["op"], "device tensor" if e["device"].startswith("cuda") else "cpu tensor")
            by[k][0] += 1
            by[k][1] += e["bytes"]
        tot = sum(v[1] for v in by.values())
        print(f"  pid {f.split('.')[-2]}: {len(ev)} ops, {tot} bytes")
        for k, v in sorted(by.items(), key=lambda kv: -kv[1][1]):
            print(f"      {k[0]:10} {k[1]:14} {v[0]:5} ops {v[1]:9} B")
        sizes = collections.Counter((e["op"], e["bytes"]) for e in ev if e["op"] in ("send", "recv"))
        print("      send/recv sizes:", dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:8]))
