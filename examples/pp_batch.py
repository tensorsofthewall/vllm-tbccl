"""Fire N concurrent greedy requests at a running server and compare each against the single-rank reference."""
import concurrent.futures as cf
import json
import sys
import urllib.request

url, model, ref_path, n_tokens, repeat = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ref = json.load(open(ref_path))
jobs = [r for _ in range(repeat) for r in ref]


def go(r):
    body = json.dumps({"model": model, "prompt": r["prompt"], "max_tokens": n_tokens, "temperature": 0, "return_token_ids": True}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=300))
    return d["choices"][0]["token_ids"] == r["token_ids"][:n_tokens]


with cf.ThreadPoolExecutor(len(jobs)) as ex:
    res = list(ex.map(go, jobs))
print(f"{sum(res)}/{len(res)} concurrent requests match the reference")
sys.exit(0 if all(res) else 1)
