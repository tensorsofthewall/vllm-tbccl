# Contributing to vllm-tbccl

Thank you for contributing. This document describes how to set up, test and submit changes. Technical rules for the code base are in `AGENTS.md`.

## Development setup

You need Python >= 3.10, vLLM (0.31.0 for CUDA/CPU, 0.30.0 for CUDA+Metal) and a matching `torch-tbccl` built against an installed TBCCL package, all in one environment with a single torch version. Use `uv`.

## Building and testing

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q tests
```

Do not use `uv pip install --reinstall`, which rewrites torch. `VLLM_TBCCL_ENABLE=1` enables the plugin in a process.

Run `tests/` for any change. Metal tests skip on Linux. End-to-end and multi-host scripts need a local model (never download weights) and are opt-in; peer-failure tests are loopback only. Say in the pull request what you ran.

## Contribution workflow

1. Open an issue for anything beyond a small fix, so the approach can be agreed first.
2. Fork or branch from `main` and keep each pull request focused on one change.
3. Make sure the build and tests pass locally and add tests for new behavior and for bug fixes.
4. Open a pull request using the template and describe what changed, why, and what you ran.
5. Maintainers review every pull request. Address review comments with follow-up commits; maintainers may ask you to squash before merging.

Project policy: changes to `main` land through pull requests, and `main` is never force-pushed. Repository settings may or may not enforce this; the policy applies either way.

## Commit messages

Use short, descriptive subjects in the form `area: summary` (imperative, no trailing period), with a body that explains why when it is not obvious. Internal tracking numbers are not required in subjects.

```text
transport: add collective data connection
protocol: validate connection roles
test: cover mixed-domain ordering
docs: document wire compatibility
ci: add documentation checks
```

## Pull request expectations

- The change builds and the relevant tests pass; state which platforms and hardware you tested on.
- Behavior changes come with tests; documentation is updated alongside code.
- No unrelated formatting or refactoring in the same pull request.
- Do not commit machine-specific paths, host names, credentials or model weights.

## Documentation

User-visible changes update the relevant documentation (`README.md` and `docs/`) in the same pull request. Document behavior that exists, not behavior you intend to add. Build the documentation with `make docs` before submitting; warnings fail the build and the same check runs in continuous integration.

## Compatibility requirements

- **Supported combinations:** vLLM versions are listed in `README.md` and enforced by `SUPPORTED_VLLM` in `vllm_tbccl/platform.py`. Adding one needs validation evidence.
- **No in-place patches** to vLLM or vllm-metal; the vllm-metal change is a patch file under `patches/`.
- **Dependencies:** requires `torch-tbccl` and, through it, an installed TBCCL package.

## AI-assisted contributions

AI-assisted contributions are permitted. Contributors remain responsible for the correctness, licensing, testing, and review of their submissions. Material AI assistance should be disclosed according to the project's contribution guidelines: add trailers to the commit message, for example

```text
AI-Assisted-By: <tool or assistant>
AI-Assistance: documentation | tests | benchmark tooling | build automation | mechanical | implementation
```

and use `AI-Validated-By: <tool>` only when the tool itself ran and recorded the validation the commit reports. Do not list an AI tool as an author, co-author, signer or reviewer.

## Reporting problems

Open an issue with the version, platform, how to reproduce, and the observed and expected behavior. Please do not include credentials or private network details.
