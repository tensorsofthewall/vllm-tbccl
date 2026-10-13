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
| `vllm-metal-0001-pluggable-pp-transport.patch` | [vllm-metal](https://github.com/vllm-project/vllm-metal) | **Yes, for the CUDA and Metal pairing** (vLLM 0.30.0). Not needed for the CUDA and CPU pairing. |
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
| vLLM | 0.30.0 (vllm-metal does not run on 0.31.0 yet) |
| vllm-metal | the revision the transport seam was developed against; the validation used vllm-metal commit `d8cb131` (2026-10-02) carrying the same seam |

```{important}
The patch file was generated against an earlier vllm-metal revision than the one used for validation. Against `d8cb131` every file applies cleanly **except `vllm_metal/envs.py`** (two hunks: the declarations of the two new environment switches), because vllm-metal reworked its environment handling after the patch was written. Use `git apply --reject` (below) and add the two switches to `envs.py` by hand, in the style of the neighboring entries. Refreshing the patch against the validated revision is tracked for the final release.
```

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

   If `--check` reports that only `vllm_metal/envs.py` fails (the case on the validated revision `d8cb131`), apply everything else and keep the rejected hunks for reference:

   ```sh
   git -C <vllm-metal checkout> apply --reject "$P"      # writes vllm_metal/envs.py.rej with the two failed hunks
   ```

   Then add to `vllm_metal/envs.py`, following how the file declares its other switches today: `VLLM_METAL_PP_TRANSPORT_CLS` (optional string, default unset: the built-in MLX ring) and `VLLM_METAL_DIST_BACKEND` (string, default `gloo`). Delete the `.rej` file afterwards. Any other failing file means your vllm-metal revision differs more; use the validated revision.

4. **Verify** the seam is present:

   ```sh
   python -c "from vllm_metal.distributed import PipelineTransport, MlxRingPipelineTransport; import vllm_metal.envs as e; print(e.environment_variables['VLLM_METAL_PP_TRANSPORT_CLS'])"
   ```

   and run vllm-tbccl's Metal tests on the Mac: `python -m pytest -q tests/test_metal_transport.py tests/test_backends.py`.

5. **Run.** Start the Metal node with `VLLM_TBCCL_BACKEND=metal` ([quickstart](../getting-started/quickstart.md)); vllm-tbccl then sets `VLLM_METAL_PP_TRANSPORT_CLS` to its transport class and `VLLM_METAL_DIST_BACKEND=tbccl` for you if you have not set them.

## Packaging

The patches are package data: they are in the wheel, the sdist and the release assets, so `python -m vllm_tbccl.patches` works from an installed package without a source checkout. `tests/test_patches.py` checks that they are discoverable and well-formed.
