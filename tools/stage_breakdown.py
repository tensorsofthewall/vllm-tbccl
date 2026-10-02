"""Per-stage decode/prefill breakdown from a trace run's output (same-process monotonic durations only; hosts are never compared)."""
import glob
import json
import statistics as st
import sys

root = sys.argv[1]


def load(pattern):
    fs = [f for f in glob.glob(pattern) if ".pg." not in f]
    return json.load(open(fs[0])) if fs else []


def us(e):
    return (e["done_ns"] - e["entry_ns"]) / 1e3


def med(xs):
    return round(st.median(xs), 0) if xs else None


def p(xs, f):
    xs = sorted(xs)
    return round(xs[min(int(f * len(xs)), len(xs) - 1)], 0) if xs else None


ALL = (("A", "CUDA stage0 -> Metal stage1"), ("B", "Metal stage0 -> CUDA stage1"), ("cA", "CUDA stage0 -> CPU stage1"))
sel = sys.argv[2:]
for ori, desc in ([(o, next(d for k, d in ALL if o == k or o.startswith(k + "_"))) for o in sel] if sel else ALL):
    lin, mac = load(f"{root}/{ori}/*_linux.[0-9]*.json"), load(f"{root}/{ori}/*_mac.[0-9]*.json")
    print(f"=== {ori}: {desc}")
    real_lin = [us(e) for e in lin if e["op"] == "step_execute" and us(e) > 1500]
    print(f"  Linux CUDA step_execute (>1.5ms, real steps) n={len(real_lin)} median={med(real_lin)}us p25={p(real_lin, .25)} p75={p(real_lin, .75)}")
    if ori.split("_")[0] == "A":
        rv = [e for e in mac if e["op"] == "metal_recv_upstream"]
        dec = [e for e in rv if e["shape"][0] == 1]
        pre = [e for e in rv if e["shape"][0] > 1]
        fw = [us(e) for e in mac if e["op"] == "step_forward"]
        sm = [us(e) for e in mac if e["op"] == "step_sample"]
        print(f"  Metal step_forward (incl. blocking recv) median={med(fw)} p25={p(fw, .25)} p75={p(fw, .75)}; step_sample (GPU wait) median={med(sm)} p25={p(sm, .25)} p75={p(sm, .75)}")
        for name, grp in (("decode (1 row)", dec), ("prefill (>1 row)", pre)):
            if grp:
                print(f"  recv {name} n={len(grp)}: total={med([us(e) for e in grp])}us  wait-for-upstream(meta)={med([e['meta_us'] for e in grp])}  "
                      f"transfer+alloc(pg)={med([e['pg_us'] for e in grp])}  of which alloc={med([e.get('alloc_us', 0) for e in grp])}  add={med([e['add_us'] for e in grp])}  bytes={sorted(set(e['bytes'] for e in grp))[:3]}")
    if ori.split("_")[0] == "B":
        sd = [e for e in mac if e["op"] == "metal_send_upstream"]
        dec = [e for e in sd if e["shape"][0] == 1]
        fw = [us(e) for e in mac if e["op"] == "step_forward"]
        sm = [us(e) for e in mac if e["op"] == "step_sample"]
        print(f"  Metal step_forward median={med(fw)} (stage0: embed+14 layers, lazy eval happens in send) ; step_sample median={med(sm)}")
        if dec:
            print(f"  send decode n={len(dec)}: total={med([us(e) for e in dec])}us eval(GPU run)={med([e['eval_us'] for e in dec])} alias={med([e['alias_us'] for e in dec])} pg(send)={med([e['pg_us'] for e in dec])}")
    if ori.split("_")[0] == "cA":
        cp = [us(e) for e in mac if e["op"] == "step_execute" and us(e) > 1500]
        rc = [e for e in mac if e["op"] == "recv"]
        print(f"  CPU stage step_execute (>1.5ms) n={len(cp)} median={med(cp)} p25={p(cp, .25)} p75={p(cp, .75)}; recv events n={len(rc)} median={med([us(e) for e in rc])}us bytes={sorted(set(e['bytes'] for e in rc))[:4]}")
