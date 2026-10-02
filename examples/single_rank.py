"""Single-rank vLLM reference run: FP32, eager, greedy. Writes token ids/text to --out (JSON)."""
import argparse
import json

from vllm import LLM, SamplingParams

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--mem", type=float, default=None)
    a = p.parse_args()

    PROMPTS = ["The capital of France is", "In a distant future, robots and humans lived together and",
               "def fibonacci(n):\n    "]
    mem = a.mem if a.mem is not None else (0.2 if a.device in ("cpu", "metal") else 0.5)
    llm = LLM(model=a.model, dtype=a.dtype, enforce_eager=True, max_model_len=512, seed=0, gpu_memory_utilization=mem)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=a.max_tokens))
    res = [{"prompt": o.prompt, "prompt_token_ids": list(o.prompt_token_ids), "token_ids": list(o.outputs[0].token_ids),
            "text": o.outputs[0].text} for o in outs]
    for r in res:
        print(json.dumps(r))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
