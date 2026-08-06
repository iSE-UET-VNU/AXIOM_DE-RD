# KDL ParseBench engine notice

`kdl_frontier_engine.py` is derived from the standalone
`kdl_frontier_nano.py` provider in
[run-llama/ParseBench](https://github.com/run-llama/ParseBench).

ParseBench is distributed under the Apache License, Version 2.0. The vendored
engine retains the original pipeline comments and deterministic inference
logic. AXIOM-specific scheduling and output adaptation are implemented in
`kdl.py`.
