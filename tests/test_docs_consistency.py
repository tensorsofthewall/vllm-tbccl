"""Documentation contract checks: the documented environment variables, versions and independence statements match the code and the manifest."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _source_env_vars():
    names = set()
    for base in ("vllm_tbccl", "csrc"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for name in files:
                if name.endswith((".py", ".cpp", ".hpp", ".h")):
                    with open(os.path.join(dirpath, name), errors="ignore") as f:
                        names.update(re.findall(r'(?:environ(?:\.get|\.setdefault)?|getenv)\(\s*"((?:VLLM|TBCCL)_[A-Z0-9_]+)"', f.read()))
                        f.seek(0)
                        names.update(re.findall(r'environ\["((?:VLLM|TBCCL)_[A-Z0-9_]+)"\]', f.read()))
    return names


def test_environment_variables_read_by_the_code_are_documented():
    page = _read("docs", "reference", "configuration.md")
    documented = set(re.findall(r"`((?:VLLM|TBCCL)_[A-Z0-9_]+)`", page))
    missing = {v for v in _source_env_vars() if v not in documented}
    assert not missing, f"environment variables read by the code but not documented: {sorted(missing)}"


def test_documented_environment_variables_exist():
    page = _read("docs", "reference", "configuration.md")
    documented = set(re.findall(r"`((?:VLLM|TBCCL)_[A-Z0-9_]+)`", page))
    known_elsewhere = {"VLLM_USE_V2_MODEL_RUNNER", "VLLM_HOST_IP", "VLLM_TBCCL_TEST_MODEL", "TBCCL_LOCAL_ENDPOINT"}  # vLLM's own, test-only, and read by the native backend through a named constant
    stale = {v for v in documented if v not in _source_env_vars() and v not in known_elsewhere}
    assert not stale, f"documented environment variables that the code does not read: {sorted(stale)}"


def test_documented_versions_match_the_manifest():
    manifest = json.loads(_read("compatibility.json"))
    compat = _read("docs", "reference", "compatibility.md")
    install = _read("docs", "getting-started", "install.md")
    version = manifest["package"]["development_version"]
    assert version in compat and version in install
    for v in manifest["frameworks"]["vllm"]["supported"]:
        assert v in compat and v in install
    assert manifest["python"]["requires"] == ">=3.12,<3.14"
    assert "3.12 and 3.13" in compat and "3.13 only" not in compat and "3.10 or newer" not in compat
    metal = manifest["frameworks"]["vllm_metal"]
    patches = _read("docs", "guides", "patches.md")
    for text in (compat, patches):
        assert metal["revision"] in text and metal["transport_patch_sha256"] in text and metal["version"] in text
        assert metal["fork_tag"] in text and metal["fork_revision"] in text
    assert metal["fork"] == "tensorsofthewall/vllm-metal" and metal["fork_tag"] == "v0.31.0-tbccl1"
    for page in (install, _read("docs", "guides", "cuda-metal.md")):
        assert metal["fork_tag"] in page
    assert ("/mnt" + "/BigChonk") not in install and "git apply" not in install
    assert "**0.31.0**" in install and "0.30.0" not in manifest["frameworks"]["vllm"]["supported"]


def test_docs_state_the_standalone_architecture_and_no_registry_publication():
    text = "\n".join(_read("docs", *p) for p in (("concepts", "architecture.md"), ("getting-started", "install.md"), ("getting-started", "overview.md"), ("project", "ecosystem.md")))
    assert "does not use or require torch-tbccl" in text or "does not depend on torch-tbccl" in text
    assert "built on torch-tbccl" not in text and "routes communication through torch-tbccl" not in text
    install = _read("docs", "getting-started", "install.md")
    assert "test.pypi.org" not in install and "pip install --pre" not in install
    assert "not published on PyPI or TestPyPI" in install


def test_documented_patches_are_the_shipped_patches():
    patches = os.listdir(os.path.join(ROOT, "vllm_tbccl", "patches"))
    page = _read("docs", "guides", "patches.md")
    for name in (p for p in patches if p.endswith(".patch")):
        assert name in page, f"{name} ships in the package but the patch workflow page does not mention it"
