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

The codebase is under active development and is not yet intended for production deployment.

Current work includes:

* Cleaning and validating inference benchmark data
* Training a lightweight regression-based policy engine
* Selecting efficient GPU clock speeds dynamically
* Testing across multiple 3B and 7B models
* Improving reproducibility across NVIDIA GPUs
* Limiting throughput loss while improving tokens per joule

Requirements

AVIS currently requires:

* Linux
* NVIDIA GPU
* NVIDIA drivers
* NVML
* CMake
* C and C++ compiler
* Git
* A compatible GGUF model
* llama.cpp

The project has primarily been tested on a consumer NVIDIA GPU.

llama.cpp Dependency

llama.cpp is not included in this repository.

Clone llama.cpp inside the AVIS Inference working directory:

git clone https://github.com/ggml-org/llama.cpp.git llama.cpp

The expected directory structure is:

Avis_Inference/
├── llama.cpp/
├── models/
├── src/
├── ML_Policy_Engine/
├── CMakeLists.txt
├── prompts.txt
└── README.md

Because the llama.cpp API changes over time, AVIS may require a specific compatible version.

A tested commit hash should be documented in this repository when compatibility is finalized.

Models

Model files are not included in this repository.

Users must download their own compatible GGUF models and place them inside the models/ directory.

Create the directory with:

mkdir -p models

Move a downloaded model into it:

mv your-model.gguf models/

Example structure:

models/
├── qwen2.5-3b-instruct-q4_k_m.gguf
└── qwen2.5-coder-7b-instruct-q4_k_m.gguf

Users are responsible for following the license and usage conditions of each model they download.

Build Instructions

Clone the AVIS repository:

git clone https://github.com/goutham-226/AVIS_ENERGY_AWARE_INFERENCE.git
cd AVIS_ENERGY_AWARE_INFERENCE

Clone llama.cpp inside the repository:

git clone https://github.com/ggml-org/llama.cpp.git llama.cpp

Create a build directory and configure the project:

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

Build the project:

cmake --build build -j"$(nproc)"

The executable location depends on the target name defined in CMakeLists.txt.

For example:

./build/avis

NVIDIA Permissions

Some GPU clock controls may require elevated permissions.

To inspect your GPU:

nvidia-smi

To continuously monitor GPU activity:

watch -n 1 nvidia-smi

To view useful inference metrics:

nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem --format=csv

Clock-setting commands may require sudo and may not be supported on every NVIDIA GPU.

Metrics

AVIS reports several efficiency and performance metrics.

Tokens per second

Tokens per second measures inference throughput:

tokens per second = generated tokens / latency

Higher values indicate faster generation.

Tokens per joule

Tokens per joule measures energy efficiency:

tokens per joule = generated tokens / total energy

Higher values indicate that more tokens were generated for each joule of energy consumed.

Joules per token

Joules per token measures the energy required to generate one token:

joules per token = total energy / generated tokens

Lower values indicate better energy efficiency.

Throughput decrease

AVIS may calculate the percentage decrease in tokens per second compared with a baseline clock:

throughput decrease (%) =
(baseline tokens/s - current tokens/s)
/
baseline tokens/s
× 100

Energy-efficiency increase

AVIS may calculate the percentage increase in tokens per joule compared with a baseline:

tokens-per-joule increase (%) =
(current tokens/J - baseline tokens/J)
/
baseline tokens/J
× 100

Experimental Method

AVIS compares multiple GPU SM clock speeds for the same inference workload.

A workload may include:

* Model size
* Prompt-token count
* Maximum output length
* Generated-token count
* Batch size
* KV-cache usage
* GPU clock speed
* Inference latency
* Energy consumption
* Tokens per second
* Tokens per joule

The highest tested clock is used as the baseline for each comparable workload group.

Measurements should be repeated because GPU power and latency can vary due to:

* GPU temperature
* Background processes
* Clock fluctuations
* Model loading
* Warm-up effects
* Power-sampling intervals
* Different generated-token counts

Iteration zero or the first run may contain warm-up overhead and should be treated carefully.

Dataset Cleaning

AVIS benchmark data should only compare rows representing the same workload.

Rows should be grouped using matching values such as:

* Model size
* Prompt length
* Maximum output length
* Batch size
* KV-cache configuration
* Generation settings

When generated-token counts differ, raw total energy should not be compared by itself.

Prefer normalized metrics such as:

* Tokens per joule
* Joules per token
* Tokens per second

Potentially noisy or invalid rows include:

* Lower-clock runs that consume more energy than the baseline
* Runs with unexpectedly different generated-token counts
* Runs with very small energy savings but large throughput loss
* Thermal throttling events
* Failed or interrupted inference runs
* Model-loading or warm-up measurements

Removed rows should be stored in a separate audit dataset rather than deleted permanently.

Policy Engine

The planned AVIS policy engine will use workload characteristics to predict an efficient GPU operating configuration.

Potential inputs include:

* Model size
* Prompt-token count
* Requested output length
* Batch size
* KV-cache size or usage
* GPU temperature
* Baseline latency
* GPU clock speed

Potential outputs include:

* Recommended SM clock
* Expected tokens per second
* Expected tokens per joule
* Expected energy savings
* Expected performance loss

The first version is intended to use a lightweight regression model that can run on the CPU without consuming GPU resources needed for inference.

Repository Structure

Avis_Inference/
├── src/
│   └── main.cpp
├── ML_Policy_Engine/
│   ├── inputs.txt
│   └── training_data_gen.cpp
├── models/
├── llama.cpp/
├── CMakeLists.txt
├── compile.c
├── data_compile.c
├── prompts.txt
└── README.md

The models/ and llama.cpp/ directories are expected locally but are excluded from Git.

Files Not Included

The following files are intentionally excluded from this repository:

* llama.cpp source
* GGUF model files
* Generated build files
* Compiled binaries
* Logs
* Large benchmark datasets
* IDE configuration files
* Environment files

Limitations

AVIS currently has several limitations:

* NVIDIA GPUs only
* Depends on NVML
* GPU clock control varies by GPU and driver
* Results may not generalize across different GPU architectures
* Model quantization can affect performance and energy behavior
* Calibration may be required for each GPU
* The current policy engine is still under development
* The project is not production-ready

Safety and System Notes

Changing GPU clocks can cause instability if unsupported values are used.

Users should:

* Verify supported clocks before applying changes
* Monitor temperature and power usage
* Avoid changing clocks during unrelated workloads
* Restore default clock behavior after testing
* Use administrative privileges carefully

AVIS is provided for experimentation and research.

Roadmap

Planned work includes:

* Add a trained regression policy engine
* Support automatic workload calibration
* Improve prefill and decode phase detection
* Add structured CSV benchmark output
* Add dataset validation and noise detection
* Test more 3B and 7B GGUF models
* Add reproducible benchmark scripts
* Add GPU temperature-aware decisions
* Add configurable latency-loss limits
* Add automated comparison against baseline clocks
* Improve CMake portability
* Add tests and continuous integration
* Document supported llama.cpp commits

Contributing

AVIS Inference is currently an early-stage personal research project.

Bug reports, benchmark results, reproducibility feedback, and technical suggestions are welcome.

When reporting results, include:

* GPU model
* Driver version
* Operating system
* Model name
* Quantization
* Prompt-token count
* Generated-token count
* Batch size
* Clock speed
* Latency
* Energy consumption
* Tokens per second
* Tokens per joule

License

A project license has not yet been finalized.

Before redistributing or using AVIS commercially, review the licenses of:

* AVIS Inference
* llama.cpp
* The GGUF model being used
* Any additional dependencies

Disclaimer

AVIS Inference is experimental software.

Results may vary across GPUs, models, drivers, temperatures, and workloads. The project makes no guarantee of energy savings, stability, performance, or hardware compatibility.

Platform Support

AVIS Inference is intended to support:

* Linux with an NVIDIA GPU
* Windows 10 or Windows 11 with an NVIDIA GPU

The project is currently experimental. Linux is the primary development environment, and Windows support may require additional testing.

AVIS depends on NVIDIA CUDA, NVML, CMake, and llama.cpp. These dependencies are available on both Linux and Windows.

Requirements

Common requirements

* NVIDIA CUDA-capable GPU
* Compatible NVIDIA driver
* CUDA Toolkit
* NVML
* CMake
* Git
* C++17-compatible compiler
* Compatible GGUF model
* llama.cpp

Model files and the llama.cpp source are not included in this repository.

Linux Setup

Install common build tools on Ubuntu or Debian:

sudo apt update
sudo apt install -y build-essential cmake git pkg-config

Verify the NVIDIA driver:

nvidia-smi

Install the CUDA Toolkit using NVIDIA’s official installation instructions for your Linux distribution.

Verify CUDA:

nvcc --version

Verify NVML:

ldconfig -p | grep nvidia-ml

Windows Setup

1. Install the NVIDIA driver

Install the latest compatible NVIDIA driver for your GPU.

Verify the installation in PowerShell:

nvidia-smi

2. Install Visual Studio Build Tools

Install Visual Studio 2022 or the Visual Studio Build Tools.

During installation, select:

Desktop development with C++

Ensure that the following components are installed:

* MSVC C++ compiler
* Windows SDK
* CMake tools for Windows

3. Install CMake

Install CMake and enable the option to add CMake to the system PATH.

Verify it in PowerShell:

cmake --version

4. Install Git

Install Git for Windows.

Verify it:

git --version

5. Install the CUDA Toolkit

Download and install the CUDA Toolkit for Windows from NVIDIA.

Verify the installation:

nvcc --version

The CUDA Toolkit installer normally includes the NVML development files required by AVIS.

Verify that NVML is available:

where.exe nvml.dll

A common runtime location is:

C:\Windows\System32\nvml.dll

The NVML header may be located inside the NVIDIA installation directory, for example:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\include\nvml.h

6. Clone AVIS

git clone https://github.com/goutham-226/AVIS_ENERGY_AWARE_INFERENCE.git
cd AVIS_ENERGY_AWARE_INFERENCE

7. Clone llama.cpp

llama.cpp is not included in this repository.

Clone it inside the AVIS working directory:

git clone https://github.com/ggml-org/llama.cpp.git llama.cpp

8. Add models

Model files are not included.

Create the model directory:

New-Item -ItemType Directory -Force models

Place compatible GGUF model files inside:

AVIS_ENERGY_AWARE_INFERENCE\models\

9. Build on Windows

Open Developer PowerShell for Visual Studio 2022 in the AVIS directory.

Configure the project:

cmake -S . -B build -DGGML_CUDA=ON

Build the Release version:

cmake --build build --config Release -j

The executable will normally be inside a Release directory, depending on the CMake target:

.\build\Release\avis.exe

Expected Directory Structure

AVIS_ENERGY_AWARE_INFERENCE/
├── llama.cpp/
├── models/
├── src/
├── ML_Policy_Engine/
├── CMakeLists.txt
└── README.md

Platform Notes

Linux and Windows use different compilers, library paths, executable names, and privilege models.

Avoid hardcoded paths such as:

/home/goutham/...

or:

C:\Users\Goutham\...

Use CMake variables and relative paths instead.

Linux binaries normally have names such as:

avis

Windows binaries normally have names such as:

avis.exe

Linux may require sudo for GPU clock-control commands. On Windows, GPU clock-control capabilities depend on the NVIDIA driver, GPU model, and available management interfaces.

Not every NVIDIA consumer GPU allows application clock changes. AVIS should detect unsupported controls and continue with telemetry-only operation rather than terminating.s
