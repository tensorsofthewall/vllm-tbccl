"""Teacher-forced top-k logprobs at one position: POST prompt = prompt_ids + forced token ids, max_tokens=1, logprobs=k."""
import json
import sys
import urllib.request

url, model, ref_path, idx, nforced, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
r = json.load(open(ref_path))[idx]
ids = r["prompt_token_ids"] + r["token_ids"][:nforced]
body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "logprobs": k, "return_token_ids": True}).encode()
d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=120))
lp = d["choices"][0]["logprobs"]["top_logprobs"][0]
print(json.dumps({"token": d["choices"][0]["token_ids"], "top": dict(sorted(lp.items(), key=lambda kv: -kv[1])[:k])}))
