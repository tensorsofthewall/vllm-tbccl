# Testing

```sh
python -m pytest -q tests
```

Run in an environment that has vLLM, torch-tbccl and this package. Metal tests skip on Linux; end-to-end tests skip without a local copy of the model (set `PHASE48_MODEL` to its directory; nothing is downloaded). Peer-failure tests are loopback only. Real multi-host runs use the scripts in `scripts/` (for example `physical_validation.sh`, which reads the PCIe error counters through `aer_snapshot.sh` before and after each phase) and need a direct link between the hosts; they are not part of an ordinary run.

The reference fixtures that strict end-to-end checks compare against are documented in `tests/fixtures/README.md`.

`pytest` run from the repository root also collects `examples/gloo_test.py`, which needs a distributed launcher; run `pytest tests` instead.
