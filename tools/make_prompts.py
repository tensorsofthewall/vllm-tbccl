"""Build the fixed the vLLM pipeline-parallel study work prompt set: three short text prompts plus exact-length token prompts cut from a local text file."""
import argparse
import json

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--text", required=True)
p.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
p.add_argument("--out", required=True)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
ids = tok.encode(open(a.text).read(), add_special_tokens=False)
out = [{"name": f"short{i}", "prompt": t} for i, t in enumerate(
    ["The capital of France is", "In a distant future, robots and humans lived together and", "def fibonacci(n):\n    "])]
for n in a.lengths:
    out.append({"name": f"p{n}", "prompt_token_ids": ids[:n]})
json.dump(out, open(a.out, "w"))
print(a.out, [(o["name"], len(o.get("prompt_token_ids") or tok.encode(o["prompt"]))) for o in out])
