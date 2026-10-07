# Security model

vllm-tbccl is an adapter over TBCCL and has the same trust model: **trusted peers on a trusted network**. There is no peer authentication, no transport encryption, no message authentication and no authorization. Read the TBCCL security model first; this page lists what vllm-tbccl adds.

vllm-tbccl carries vLLM's device-group traffic and, in the Metal pairing, its pipeline activations over torch-tbccl and TBCCL, so activations and intermediate tensors cross the link in clear text. Endpoint exchange uses `torch.distributed` rendezvous and vLLM's own distributed-initialization addresses, which are also unauthenticated. The vLLM HTTP API server is vLLM's and is unaffected by this plugin: it has its own security considerations that the plugin does not change.

The plugin loads no data from the peer other than tensors handled by torch-tbccl; it does not unpickle or execute anything received from a peer.

## Deployment assumptions

- All ranks are run by the same trusted party, on hosts and links that no untrusted party can reach: loopback, a private network, or a direct link such as Thunderbolt.
- If traffic must cross a network that is not trusted, put it inside a tunnel that provides authentication and encryption (a VPN or an encrypted overlay). vllm-tbccl does not provide one.
- Do not expose rank listeners to the public internet. Bind to the specific address of the trusted link rather than a wildcard where the interface allows it.

## Reporting a vulnerability

See the security policy in the repository (`SECURITY.md`). A clean error on malformed input is a robustness property and does not mean the component is safe against a hostile peer.
