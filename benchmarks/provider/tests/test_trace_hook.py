"""The diagnostic timing hook (offline): it does nothing without its environment variable, and with it installs one import finder and never raises when vLLM is absent."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "trace_hook"


class HookInert(unittest.TestCase):
    def test_the_timing_hook_does_nothing_without_its_environment_variable(self):
        code = "import sys; before = list(sys.meta_path); import sitecustomize; print(sys.meta_path == before)"
        env = {k: v for k, v in os.environ.items() if k != "TBCCL_B5_TRACE_DIR"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HOOK), env=env)
        self.assertEqual(out.stdout.strip(), "True", out.stderr)

    def test_the_timing_hook_installs_one_import_finder_when_enabled_and_never_raises_without_vllm(self):
        with tempfile.TemporaryDirectory() as d:
            code = "import sys; n = len(sys.meta_path); import sitecustomize; print(len(sys.meta_path) - n)"
            env = {**os.environ, "TBCCL_B5_TRACE_DIR": d}
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HOOK), env=env)
            self.assertEqual(out.stdout.strip(), "1", out.stderr)


if __name__ == "__main__":
    unittest.main()
