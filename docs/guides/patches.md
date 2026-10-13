# The patch workflow

vllm-tbccl is not a vLLM fork. vLLM is used unmodified: every adaptation uses a public extension point (the platform plugin entry point, `worker_cls`, a scoped `new_group` wrapper). The vllm-metal seam is published as the immutable fork tag `tensorsofthewall/vllm-metal@v0.31.0-tbccl1`; patch files ship **inside the wheel and the sdist** for audit and for cases where an upstream project would need a change that does not belong in vllm-tbccl itself.

## What ships

```sh
python -m vllm_tbccl.patches --list                         # names of the shipped patches
python -m vllm_tbccl.patches --path                         # the patch directory
python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch    # one file
```

| Patch | Applies to | Needed? |
|---|---|---|
| `vllm-metal-0001-pluggable-pp-transport.patch` | [vllm-metal](https://github.com/vllm-project/vllm-metal) | **Reference copy only.** Users install the fork tag `tensorsofthewall/vllm-metal@v0.31.0-tbccl1`, which already contains this change; no patch step. Not needed for the CUDA and CPU pairing. |
| `0001-generic-heterogeneous-hooks.patch` | vLLM 0.30.0 | **No.** A legacy patch kept for history. vLLM 0.30.0 and 0.31.0 run unmodified with the plugin; do not apply it. |

## What the vllm-metal patch does

It adds a generic, backwards-compatible seam to vllm-metal's pipeline parallelism, with unchanged default behavior:

- a `PipelineTransport` protocol and a default `MlxRingPipelineTransport` that wraps the existing MLX ring;
- the environment switches `VLLM_METAL_PP_TRANSPORT_CLS` (a transport class by qualified name) and `VLLM_METAL_DIST_BACKEND` (the `torch.distributed` backend of the worker's control groups, default `gloo`);
- an optional `begin_step` hook that tells the transport the packed row layout of each step.

vllm-tbccl supplies a transport class that sends the activations over TBCCL. Without the seam, vllm-metal has no way to take that transport. The change is meant to be proposed upstream separately, after which this step disappears.

## Which versions it is for

| | |
|---|---|
| vLLM | 0.31.0 |
| vllm-metal fork | `tensorsofthewall/vllm-metal`, tag `v0.31.0-tbccl1`, commit `fa4f934586bc27ac974a54183237594d1dc28fc5` |
| upstream vllm-metal | **0.31.0**, commit `e7a0c90bec180e5dd4275c473fd7a36e709a930e` (tag `v0.31.0`, 2026-10-12) |
| reference patch SHA-256 | `af950dfc4da283efcbbd7e55dcf28d1ffe6d7d608ded034a6888ea56844b3756` |

The fork tag is exactly upstream `v0.31.0` plus one commit: nine files (+293/-12) adding the seam, focused tests (`tests/test_pipeline_transport.py`, two environment-variable cases in `tests/test_envs.py`) and the two switches in `docs/configuration.md`. The reference patch is `git format-patch` of that commit and applies to the upstream revision with `git apply --check` and `git am`. With the seam applied vllm-metal's own test-suite results are unchanged: the failing tests are the ones that need model downloads, and they fail the same way on a clean upstream checkout. The tag is immutable; a refresh for a newer vllm-metal is published as a new tag, never by moving this one.

## Install the fork, or inspect and apply the reference patch

Normal users only install the fork tag ([install](../getting-started/install.md)). The remaining steps are for auditing the change or carrying it onto your own vllm-metal checkout.

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
