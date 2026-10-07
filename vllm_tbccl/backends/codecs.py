"""Stage-boundary codecs between an upstream vLLM stage and a vllm-metal stage.

Upstream vLLM passes IntermediateTensors{hidden_states, residual} between pipeline stages; the true residual stream is their sum
(the next layer's fused add+RMSNorm computes exactly that). vllm-metal (mlx-lm models) passes the residual stream itself. A codec states
that algebra for one model family. Each entry is an explicit allowlist, backed by an offline split-parity proof (tools/split_parity.py
and); an architecture without an entry is rejected rather than assumed to behave like another.
"""


class BoundaryCodec:
    name = ""
    architectures: frozenset = frozenset()
    keys = ("hidden_states", "residual")

    def to_metal(self, tensors):
        """upstream {hidden_states, residual} -> the single Metal stream array."""
        return tensors["hidden_states"] + tensors["residual"]

    def from_metal(self, x, zeros):
        """Metal stream array -> upstream {hidden_states, residual}; residual = 0 so the next upstream layer computes x + 0."""
        return {"hidden_states": x, "residual": zeros}


class LlamaBoundaryCodec(BoundaryCodec):
    name = "llama"
    architectures = frozenset({"LlamaForCausalLM"})


class Qwen3BoundaryCodec(BoundaryCodec):
    name = "qwen3"
    architectures = frozenset({"Qwen3ForCausalLM"})


_CODECS = (LlamaBoundaryCodec(), Qwen3BoundaryCodec())


def codec_for(architectures) -> BoundaryCodec:
    for c in _CODECS:
        if c.architectures & set(architectures or ()):
            return c
    raise NotImplementedError(
        f"vllm-tbccl: no proven CUDA/CPU<->Metal boundary codec for architectures {list(architectures or [])}; "
        f"supported: {sorted(a for c in _CODECS for a in c.architectures)}")
