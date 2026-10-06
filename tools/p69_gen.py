"""Phase 69: deterministic generation client for a vLLM OpenAI-compatible server: greedy completions with top-5 logprobs for a fixed prompt set; records prompt/generated token
text, per-position chosen logprob and top-2 margin, so strict controls (every margin comfortably non-zero) can be chosen and compared across devices / pipeline layouts.

    python tools/p69_gen.py --api http://127.0.0.1:8168 --model <path> --out FILE.json [--tokens 16] [--prompts all|name,name]
"""
import argparse
import json
import urllib.request

PROMPTS = {
    "capital": "The capital of France is",
    "count": "Count from one to ten: one, two, three,",
    "alphabet": "A B C D E F G H I J K L M N O",
    "python": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "days": "Monday, Tuesday, Wednesday, Thursday,",
    "numbers": "1, 2, 3, 4, 5, 6, 7, 8,",
    "water": "Water is made of hydrogen and",
    "medium": "Summarize the following in two sentences. " + " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog number {i * 3}." for i in range(1, 9)),
}


def call(api, model, prompt, n):
    req = urllib.request.Request(api + "/v1/completions", method="POST", headers={"content-type": "application/json"},
                                 data=json.dumps({"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": 5}).encode())
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompts", default="all")
    a = ap.parse_args()
    names = list(PROMPTS) if a.prompts == "all" else a.prompts.split(",")
    out = {}
    for n in names:
        d = call(a.api, a.model, PROMPTS[n], a.tokens)
        c = d["choices"][0]
        lp = c["logprobs"]
        margins = []
        for chosen, top in zip(lp["token_logprobs"], lp["top_logprobs"]):
            vals = sorted(top.values(), reverse=True)
            margins.append(round(vals[0] - vals[1], 4) if len(vals) > 1 else None)
        out[n] = {"prompt_tokens": d["usage"]["prompt_tokens"], "text": c["text"], "tokens": lp["tokens"], "logprobs": [round(x, 4) for x in lp["token_logprobs"]], "margins": margins,
                  "min_margin": min(m for m in margins if m is not None)}
        print(f"{n:9} min margin {out[n]['min_margin']:7.3f}  {out[n]['text'][:60]!r}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
