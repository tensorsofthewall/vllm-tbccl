"""Sequential greedy requests against a running server; prints median/p10/p90 wall time per request and per generated token."""
import json, statistics as st, sys, time, urllib.request
url, model, n, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompt = "In a distant future, robots and humans lived together and"
def one():
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}).encode()
    t = time.monotonic()
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=300))
    return time.monotonic() - t, d["usage"]["completion_tokens"]
one(); one()
xs = [one() for _ in range(reps)]
ts = sorted(t for t, _ in xs); toks = xs[0][1]
print(json.dumps({"tokens": toks, "median_s": round(st.median(ts), 4), "p10_s": round(ts[len(ts) // 10], 4), "p90_s": round(ts[-1 - len(ts) // 10], 4),
                  "ms_per_token_median": round(1e3 * st.median(ts) / toks, 2)}))
