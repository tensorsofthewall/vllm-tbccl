# Building and testing

Copy-pasteable commands. Run them from the repository root, in an environment that has vLLM and one torch version (use `uv`; never `--reinstall`, which rewrites torch). torch-tbccl must **not** be installed.

## Build the native module

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install vllm==0.31.0 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
TBCCL_ROOT=<installed tbccl prefix> uv pip install --no-build-isolation --no-deps -e .
```

## Tests

```sh
python -m pytest -q tests        # not the repository root: examples/gloo_test.py needs a distributed launcher
python -m pytest -q tests/test_independence.py tests/test_patches.py tests/test_docs_consistency.py     # the gates: standalone, patches shipped, docs match the code
```

Metal tests skip on Linux. End-to-end tests skip without a local copy of the model: set `VLLM_TBCCL_TEST_MODEL` to its directory (nothing is downloaded). Peer-failure tests are loopback only. Real multi-host runs use the scripts in `scripts/` (for example `physical_validation.sh`, which reads the PCIe error counters through `aer_snapshot.sh` before and after each phase) and need a direct link between the hosts; they are not part of an ordinary run. The reference fixtures that strict end-to-end checks compare against are documented in `tests/fixtures/README.md`.

## Wheel build and inspection

A local development wheel and sdist (the Linux wheel has a plain `linux_x86_64` tag and is not publishable):

```sh
TBCCL_ROOT=<prefix> python -m build --wheel --no-isolation -o dist .
python -m build --sdist -o dist .
python tools/inspect_wheel.py dist/vllm_tbccl-*.whl --allow-local-platform    # members, metadata, no embedded machine paths, no torch-tbccl or exo-tbccl dependency, the patches present
python tools/inspect_wheel.py dist/vllm_tbccl-*.tar.gz
python -m vllm_tbccl.patches --list                          # in an environment with the wheel installed: the patches ship in the package
```

A publishable wheel is built in a manylinux_2_28 environment (Linux) or with `MACOSX_DEPLOYMENT_TARGET` set (macOS) by `TBCCL_ROOT=<prefix> tools/build_release_wheel.sh <out-dir>`, which repairs and then inspects it; CI does this for releases.

## Documentation and manifest checks

```sh
make docs              # builds docs/_build/html with warnings as errors (creates .venv-docs)
make docs-linkcheck    # checks local links and anchors
python tools/check_compatibility_manifest.py    # compatibility.json against platform.py and pyproject.toml
```

## Release artifacts

CI (`.github/workflows/release-artifacts.yml`, run by a maintainer on a tag) builds the wheels and the sdist, inspects them, installs them into clean environments and runs the tests that need only vLLM. Maintainers verify published artifacts with `SHA256SUMS`, the SPDX SBOM and `gh attestation verify` as described in `RELEASE.md`.
