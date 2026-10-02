"""Apply the one generic vLLM 0.30.0 hook vllm-tbccl needs (idempotent): let GroupCoordinator's CPU/control group backend be chosen by
VLLM_CPU_GROUP_BACKEND (default "gloo" = unchanged behavior). Needed because PyTorch's gloo does not rendezvous between the Linux and macOS
builds, so a Linux<->macOS vLLM world cannot use gloo for its control groups. See docs/vllm_api_audit.md."""
import pathlib
import sys

import vllm

path = pathlib.Path(vllm.__file__).parent / "distributed" / "parallel_state.py"
src = path.read_text()
old = '''                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks, backend="gloo", timeout=timeout
                    )'''
new = '''                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks,
                        backend=os.environ.get("VLLM_CPU_GROUP_BACKEND", "gloo"),
                        timeout=timeout,
                    )'''
if new in src:
    print("already applied:", path)
    sys.exit(0)
assert src.count(old) == 1, "unexpected vLLM source; patch not applied"
if "\nimport os\n" not in src:
    src = src.replace("\nimport pickle\n", "\nimport os\nimport pickle\n", 1)
path.write_text(src.replace(old, new))
print("patched:", path)
