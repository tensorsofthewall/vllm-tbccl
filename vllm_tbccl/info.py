"""``python -m vllm_tbccl.info``: print the versions of every layer of this installation as JSON."""
import json

from ._backend import info

if __name__ == "__main__":
    print(json.dumps(info(), indent=2))
