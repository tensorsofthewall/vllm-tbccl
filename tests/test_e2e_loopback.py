"""The real vLLM 0.31.0 serving engine, PP=2 / TP=1, two worker processes on one host, every pipeline tensor and control message through vllm-tbccl / ProcessGroupTBCCL.

Skipped without a local Qwen3-0.6B (``~/models/Qwen3-0.6B`` or ``$PHASE48_MODEL``); nothing is downloaded. Strict deterministic controls are the prompts whose argmax margin
is comfortably non-zero on both devices (tests/fixtures/reference_*.json); near-tie prompts are deliberately not used as correctness controls.
"""
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("PHASE48_MODEL", os.path.expanduser("~/models/Qwen3-0.6B"))
HAS_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL), reason="needs the local Qwen3-0.6B (never downloaded)")
REF = json.load(open(os.path.join(ROOT, "tests/fixtures/reference_vllm031_linux_cuda.json")))
STRICT = ("numbers", "medium", "python")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import reference_generator  # noqa: E402


def _env():
    e = dict(os.environ)
    if HAS_CUDA:
        e["EXTRA_ARGS"] = "--kv-cache-memory-bytes 300000000"
    else:
        e.update(EXTRA_ENV="VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=4", EXTRA_ARGS="--kv-cache-memory-bytes 300000000", GPU_UTIL="0.2", VLLM_TBCCL_PLATFORM="cpu")
    return e


class Engine:
    def __init__(self, tag):
        self.tag, self.api = tag, "http://127.0.0.1:" + os.environ.get("API_PORT", "8169")

    def __enter__(self):
        r = subprocess.run(["bash", os.path.join(ROOT, "scripts/loopback_engine.sh"), "up", self.tag], env=_env(), capture_output=True, text=True, timeout=600)
        assert "UP after" in r.stdout, r.stdout + r.stderr + _tail(self.tag)
        return self

    def __exit__(self, *a):
        self.down()

    def pids(self):
        out = []
        for role in ("leader", "worker"):
            f = os.path.join(ROOT, "results", f"{self.tag}_{role}.pid")
            if os.path.exists(f):
                out.append(int(open(f).read()))
        return out

    def down(self):
        t0 = time.monotonic()
        subprocess.run(["bash", os.path.join(ROOT, "scripts/loopback_engine.sh"), "down", self.tag], env=_env(), timeout=60)
        deadline = t0 + 60
        while time.monotonic() < deadline and any(_alive_group(p) for p in self.pids()):
            time.sleep(0.2)
        return time.monotonic() - t0


def _alive_group(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _tail(tag):
    out = ""
    for role in ("leader", "worker"):
        f = os.path.join(ROOT, "results", f"{tag}_{role}.log")
        if os.path.exists(f):
            out += f"\n--- {role} ---\n" + "".join(open(f).readlines()[-25:])
    return out


def _complete(api, prompt_name, n=16):
    d = reference_generator.call(api, MODEL, reference_generator.PROMPTS[prompt_name], n)
    return d["choices"][0]["logprobs"]["tokens"], d


def _assert_strict(api):
    for name in STRICT:
        toks, _ = _complete(api, name)
        assert toks == REF[name]["tokens"], (name, toks, REF[name]["tokens"])


def test_engine_pp2_strict_controls_concurrency_cancellation_and_shutdown():
    with Engine("t_e2e") as eng:
        api = eng.api
        _assert_strict(api)                                              # short / medium / code prompts, greedy, token-for-token against the single-rank reference
        _assert_strict(api)                                              # sequential repeat
        jobs = [STRICT[i % 3] for i in range(12)]                        # the vLLM pipeline-parallel integration 12-request concurrency test
        with cf.ThreadPoolExecutor(12) as ex:
            res = list(ex.map(lambda n: (n, _complete(api, n)[0]), jobs))
        assert all(toks == REF[n]["tokens"] for n, toks in res), [n for n, t in res if t != REF[n]["tokens"]]
        # cancellation: abandon a streaming request after a few chunks; the engine must keep serving
        req = urllib.request.Request(api + "/v1/completions", method="POST", headers={"content-type": "application/json"},
                                     data=json.dumps({"model": MODEL, "prompt": reference_generator.PROMPTS["count"], "max_tokens": 400, "temperature": 0, "stream": True}).encode())
        r = urllib.request.urlopen(req, timeout=60)
        for i, _line in enumerate(r):
            if i >= 3:
                break
        r.close()
        time.sleep(2)
        _assert_strict(api)
        t_down = eng.down()
    assert t_down < 40, t_down                                           # bounded shutdown, no leftover process group
    assert not any(_alive_group(p) for p in eng.pids())


def test_repeated_engine_create_destroy_leaves_nothing_behind():
    for i in range(3):
        with Engine(f"t_cyc{i}") as eng:
            _assert_strict(eng.api)
            assert eng.down() < 40
        assert not any(_alive_group(p) for p in eng.pids())
    if HAS_CUDA:
        used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])
        assert used < 1500, used                                         # the GPU memory of all three engines was returned
