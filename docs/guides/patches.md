# The patch workflow

vllm-tbccl is not a vLLM fork. vLLM is used unmodified: every adaptation uses a public extension point (the platform plugin entry point, `worker_cls`, a scoped `new_group` wrapper). Patch files ship **inside the wheel and the sdist** only for cases where an upstream project would need a change that does not belong in vllm-tbccl itself.

## What ships

```sh
python -m vllm_tbccl.patches --list                         # names of the shipped patches
python -m vllm_tbccl.patches --path                         # the patch directory
python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch    # one file
```

| Patch | Applies to | Needed? |
|---|---|---|
| `vllm-metal-0001-pluggable-pp-transport.patch` | [vllm-metal](https://github.com/vllm-project/vllm-metal) | **Yes, for the CUDA and Metal pairing** (vllm-metal 0.31.0 with vLLM 0.31.0). Not needed for the CUDA and CPU pairing. |
| `0001-generic-heterogeneous-hooks.patch` | vLLM 0.30.0 | **No.** A legacy patch kept for history. vLLM 0.30.0 and 0.31.0 run unmodified with the plugin; do not apply it. |

## What the vllm-metal patch does

It adds a generic, backwards-compatible seam to vllm-metal's pipeline parallelism, with unchanged default behavior:

- a `PipelineTransport` protocol and a default `MlxRingPipelineTransport` that wraps the existing MLX ring;
- the environment switches `VLLM_METAL_PP_TRANSPORT_CLS` (a transport class by qualified name) and `VLLM_METAL_DIST_BACKEND` (the `torch.distributed` backend of the worker's control groups, default `gloo`);
- an optional `begin_step` hook that tells the transport the packed row layout of each step.

vllm-tbccl supplies a transport class that sends the activations over TBCCL. Without the patch, vllm-metal has no way to take that transport. The patch is meant to be proposed upstream separately, after which this step disappears.

## Which versions it is for

| | |
|---|---|
| vLLM | 0.31.0 |
| vllm-metal | **0.31.0**, commit `e7a0c90bec180e5dd4275c473fd7a36e709a930e` (tag `v0.31.0`, 2026-10-12) |
| patch SHA-256 | `f0a13f4874dbc7b95973f5e237cdd4b7d02108597dcc707b51cc95ca04c87b07` |

The patch was generated with `git format-patch` from a commit on top of that exact vllm-metal revision and applies to it with `git apply --check` and `git am` without changes. It adds focused tests (`tests/test_pipeline_transport.py`, two environment-variable cases in `tests/test_envs.py`) and documents the two switches in vllm-metal's `docs/configuration.md`; with the patch applied vllm-metal's own test-suite results are unchanged (the same 44 tests that need model downloads fail on a clean checkout and on the patched one). It targets one exact upstream revision: on a later vllm-metal it may need refreshing, and `git apply --check` tells you.

## Inspect, apply, verify

1. **Inspect.** The patch is a mail-formatted git patch; read it, and see which files it touches:

   ```sh
   P="$(python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch)"
   git apply --stat "$P"
   ```

2. **Check that it applies** to your vllm-metal checkout (no changes are made):

   ```sh
   git -C <vllm-metal checkout> apply --check "$P"
   ```

3. **Apply** it. If `--check` was clean, apply it as a commit (an editable install of the checkout picks it up):

   ```sh
   git -C <vllm-metal checkout> am "$P"
   ```

   If `--check` fails, your vllm-metal differs from `v0.31.0`; check out the tag (`git checkout v0.31.0`) rather than editing the patch by hand. A refreshed patch for a newer vllm-metal is produced from a commit on top of that revision with `git format-patch`.

4. **Verify** the seam is present:

   ```sh
   python -c "from vllm_metal.distributed import PipelineTransport, MlxRingPipelineTransport; import vllm_metal.envs as e; print(e.environment_variables['VLLM_METAL_PP_TRANSPORT_CLS'])"
   ```

   and run vllm-tbccl's Metal tests on the Mac: `python -m pytest -q tests/test_metal_transport.py tests/test_backends.py`.

5. **Run.** Start the Metal node with `VLLM_TBCCL_BACKEND=metal` ([quickstart](../getting-started/quickstart.md)); vllm-tbccl then sets `VLLM_METAL_PP_TRANSPORT_CLS` to its transport class and `VLLM_METAL_DIST_BACKEND=tbccl` for you if you have not set them.

## Packaging

The patches are package data: they are in the wheel, the sdist and the release assets, so `python -m vllm_tbccl.patches` works from an installed package without a source checkout. `tests/test_patches.py` checks that they are discoverable and well-formed.
