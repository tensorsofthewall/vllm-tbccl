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


def test_documented_wheel_urls_name_the_current_release():
    manifest = json.loads(_read("compatibility.json"))
    version = manifest["package"]["development_version"]
    for name in (("README.md",), ("docs", "getting-started", "install.md")):
        text = _read(*name)
        urls = re.findall(r"https://github\.com/tensorsofthewall/vllm-tbccl/releases/download/\S+?\.whl", text)
        assert urls, name
        for url in urls:
            assert f"/download/v{version}/vllm_tbccl-{version}-cp312-cp312-" in url, url
        assert any("manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" in u for u in urls)
        assert any("macosx_14_0_arm64.whl" in u for u in urls)


def test_install_commands_name_the_environment_and_pin_the_fork():
    manifest = json.loads(_read("compatibility.json"))
    fork_tag = manifest["frameworks"]["vllm_metal"]["fork_tag"]
    for name in (("README.md",), ("docs", "getting-started", "install.md")):
        text = _read(*name)
        wheel_lines = [ln for ln in text.splitlines() if "uv pip install" in ln and "vllm_tbccl-" in ln and ".whl" in ln]
        assert wheel_lines
        for ln in wheel_lines:
            assert "--no-deps" in ln and "--python " in ln, ln
        assert "--torch-backend cu130" in text and f"git clone --branch {fork_tag} --depth 1" in text
        assert "source .venv/bin/activate" not in text and ". .venv/bin/activate" not in text and "source .venv-vllm-metal" not in text


def test_readme_is_free_of_source_build_instructions():
    assert "TBCCL_ROOT" not in _read("README.md") and "pip install -e" not in _read("README.md")


def test_quickstart_commands_use_variables_not_one_machines_values():
    text = _read("docs", "getting-started", "quickstart.md")
    commands = "\n".join(re.findall(r"```sh\n(.*?)```", text, re.S))
    assert "192.168.3." not in commands
    for name in ("LIP", "MIP", "MODEL"):
        assert re.search(rf"^{name}=", commands, re.M), name
    assert "--master-addr $LIP" in commands and "--host $LIP" in commands
    assert commands.count("VLLM_USE_V2_MODEL_RUNNER=0") >= 2


def test_metal_install_uses_the_source_checkout_and_no_expiring_artifact():
    text = _read("docs", "getting-started", "install.md")
    assert "no prebuilt public wheel of the fork" in text
    assert "git clone --branch v0.31.0-tbccl1" in text
    for doc in ("install.md", "quickstart.md"):
        body = _read("docs", "getting-started", doc).lower()
        assert "actions artifact" not in body and "nightly.link" not in body and "workflow run" not in body
