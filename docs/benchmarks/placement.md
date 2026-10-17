# Pipeline placement

With pipeline parallelism over two machines, **stage 0 runs the first layers and also hosts the API server** (the node with node rank 0). Stage order therefore decides three things at once: which machine computes the first layers, which machine accepts requests, and where a sampler or a client on the API host runs. A benchmark that swaps the stage order swaps all three.

The runner calls the two cases *orientations*:

* **CUDA-first**: the Linux CUDA machine is stage 0 and hosts the API.
* **Metal-first**: the Mac is stage 0 and hosts the API.

A comparison between orientations is only valid for the same model, layer split, dtype, prompt and output lengths, concurrency, and client placement. The runner records the orientation, the stage hosts, the stage devices and where the client ran in every result, and refuses to compare results that differ in them.

The layer split between stages is the engine's default for the model. Measurements of other splits are separate experiments; `partition` settings are recorded as part of the configuration.

When two orientations differ in speed, the difference is a measured fact about that configuration; it is **not** thereby attributable to the network. In the reference measurements the boundary exchange was a small fraction of a step and the stage order, the API host and the sampler location were confounded (see [Results](results.md)).

A single-node reference (`rank_counts: [1]` with one host) gives the same workload on one device, which is the baseline a pipeline must be compared with before drawing a conclusion about distribution.
