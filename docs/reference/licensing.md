# License and third-party software

vllm-tbccl is licensed under the Apache License, Version 2.0. The full text is in the `LICENSE` file at the repository root. Copyright 2026 Sandesh Bharadwaj.

Contributions are accepted under the same license (Apache-2.0, section 5).

## Patch files

The directory `vllm_tbccl/patches/` contains two patch files. They are modifications of other projects, both licensed Apache-2.0:

| Patch | Applies to | Upstream license |
|---|---|---|
| `0001-generic-heterogeneous-hooks.patch` | vLLM | Apache-2.0 |
| `vllm-metal-0001-pluggable-pp-transport.patch` | vllm-metal | Apache-2.0 |

A patch consists of the added and changed lines plus a few lines of unchanged upstream context. The added code is contributed under the Apache License 2.0, the same license as the project it modifies; the unchanged context lines remain the upstream project's. A patched checkout is a modified copy of the upstream project and stays under its Apache-2.0 license, including its copyright notices. At the audited revisions neither vLLM nor vllm-metal has a `NOTICE` file, so there is no upstream notice to carry. The patches are not part of the installed wheel.

## The package

vllm-tbccl does not vendor or copy source from vLLM, vllm-metal or torch-tbccl. It declares them as dependencies and uses them through their public interfaces.

| Dependency | Role | License |
|---|---|---|
| vLLM | runtime dependency | Apache-2.0 |
| vllm-metal | optional, Metal pairing | Apache-2.0 |
| torch-tbccl | runtime dependency | Apache-2.0 |
| PyTorch | runtime dependency | BSD-style |

## Notices

No third-party `NOTICE` file or license bundle is required. A `NOTICE` file is not provided.
