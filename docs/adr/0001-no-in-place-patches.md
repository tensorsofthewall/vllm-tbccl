# ADR 0001: Extend vLLM and vllm-metal without patching them in place

- Status: accepted
- Date: 2026-10-02

## Context

vllm-tbccl has to connect vLLM stages that differ in platform (CUDA, CPU, Metal). Earlier versions applied a small patch set to vLLM 0.30.0; a current released vLLM should be usable unmodified.

## Decision

vLLM is used unmodified. Every adaptation uses a public extension point: the platform plugin entry point, the public `worker_cls` setting for thin worker subclasses, and a scoped `new_group` wrapper that is active only for callers in `vllm.distributed.parallel_state`. vllm-metal needs one generic change, a pluggable pipeline-transport seam with unchanged default behavior; it is carried as a patch file in `patches/` and is meant to be proposed upstream separately. vllm-tbccl owns no transport or algorithm: anything missing belongs to torch-tbccl or TBCCL.

## Consequences

- A supported vLLM version range must be listed and validated (`SUPPORTED_VLLM`); other versions warn.
- The Metal pairing depends on a vllm-metal checkout with the patch applied.
- The legacy vLLM hook patch kept in `patches/` is obsolete for 0.31.0 and remains only as history.
