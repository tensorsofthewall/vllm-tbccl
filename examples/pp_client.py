"""Query a running vLLM OpenAI server with the reference prompts (greedy) and compare token ids to a reference JSON."""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://192.168.3.2:8123")
p.add_argument("--model", required=True)
p.add_argument("--ref", required=True, help="reference JSON: its prompt_token_ids are the prompts; token_ids are compared")
p.add_argument("--out", default=None, help="write this run as a reference JSON (prompt_token_ids + token_ids)")
p.add_argument("--max-tokens", type=int, default=16)
p.add_argument("--timeout", type=float, default=300)
a = p.parse_args()
ref = json.load(open(a.ref))
ok = True
res = []
for r in ref:
    body = json.dumps({"model": a.model, "prompt": r["prompt_token_ids"], "max_tokens": a.max_tokens, "temperature": 0,
                       "return_token_ids": True}).encode()
    t = time.monotonic()
    d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=a.timeout))
    ch = d["choices"][0]
    same = ch["token_ids"] == r["token_ids"][: a.max_tokens] and ch["prompt_token_ids"] == r["prompt_token_ids"]
    ok &= same
    res.append({"prompt_token_ids": ch["prompt_token_ids"], "token_ids": ch["token_ids"]})
    print(f"{'MATCH' if same else 'DIFF '} {time.monotonic() - t:6.2f}s prompt_tokens={len(ch['prompt_token_ids'])} -> {ch['token_ids']}")
print("ALL MATCH" if ok else "MISMATCH")
if a.out:
    json.dump(res, open(a.out, "w"))
