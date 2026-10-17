# Methodology and limitations

* **Independent unit.** The replication unit is an independent engine lifecycle (a fresh start of the engines, with clean-up verified on both hosts). Requests inside a lifecycle are samples of that lifecycle, not replicates. Tests and confidence statements use lifecycle medians.
* **Actual tokens.** Token counts come from what the server reported. A configured maximum is never used as an output length.
* **Correctness outside the timer.** Reference and check gates run before the timed phases, with the same prompts. A run that fails a gate has no measurement.
* **Clocks.** Each interval uses one machine's monotonic clock. Timestamps of different machines are never subtracted, so a client on the other machine measures a different thing from a client on the API host.
* **Warm-up.** Warm-up requests are kept, flagged and excluded from the timed statistics.
* **Stage orientation.** Compared runs must have the same stage hosts, devices, layer split and client placement. See [Pipeline placement](placement.md).
* **Hardware observability.** Link state and interface counters are checked around each lifecycle, and the host must be idle before it starts. Where PCIe AER is disabled, the result says AER was unavailable instead of reporting zero errors.
* **Distribution.** A pipeline across two machines is evaluated against the single-device run of the same model. A pipeline slower than one device is expected when the model fits on one.
* **Not covered.** Model quality, long contexts, tensor parallelism, more than two stages, mixed-precision variants, and the physical link rate (these benchmarks run over TCP).
