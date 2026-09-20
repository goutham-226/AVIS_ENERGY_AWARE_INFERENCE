AVIS Inference

AVIS Inference is an experimental energy-aware LLM inference runtime built around llama.cpp.

The project measures inference latency, GPU power usage, energy consumption, token throughput, and tokens per joule. Its goal is to identify more energy-efficient GPU operating configurations while keeping performance loss within acceptable limits.

AVIS currently focuses on 4-bit GGUF models, especially 3B and 7B parameter models running on NVIDIA GPUs.

Current Features

* Runs GGUF models through llama.cpp
* Measures total inference latency
* Measures GPU power usage through NVML
* Calculates total energy consumption in joules
* Calculates tokens per second
* Calculates tokens per joule
* Supports GPU SM clock experiments
* Separates prefill and decode measurements
* Generates training data for an energy-aware policy engine
* Supports prompt length, output length, batch size, KV-cache usage, model size, and clock speed as experimental variables

Project Status

AVIS Inference is currently an experimental research prototype.

##Future Update:
##---------------
Avis will we be re-written in RUST for more memory safety, The source code will be modularized,
the emphasis will be on writing clean code with best practices, the inference engine will have it's own HTTP layer
written in python to upload and run AI models and get gpu telemetry. For testing, the engine will be built for a 4 gpu  node 
with tensor parallelism to allow for high-throughput energy efficient serving using intra-node disaggregation.
