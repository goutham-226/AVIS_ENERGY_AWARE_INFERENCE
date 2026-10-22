cmake_minimum_required(VERSION 3.28)
project(avis VERSION 0.2.0 DESCRIPTION "Energy-aware LLM inference engine" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AVIS_BUILD_TESTS "Build the unit and integration tests" ON)
option(AVIS_BUILD_BENCHMARKS "Build the Google Benchmark micro-benchmarks" ON)
option(AVIS_WITH_LLAMA "Build the llama.cpp backend and the avis CLI" ON)
option(AVIS_WITH_NVML "Build NVML GPU telemetry and clock control" ON)

# scripts/setup_deps.sh installs llama.cpp here.
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third_party/install")

add_library(avis_warnings INTERFACE)
target_compile_options(avis_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
                              -Wno-missing-field-initializers>)

# ---------------------------------------------------------------------------
# Core: scheduling, paged KV accounting, energy policy, telemetry interfaces.
# No dependency on llama.cpp or NVML so it can be tested anywhere.
# ---------------------------------------------------------------------------
add_library(avis_core
    src/kv/block_manager.cpp
    src/engine/scheduler.cpp
    src/engine/clock_policy.cpp
    src/engine/engine.cpp
    src/telemetry/power_sampler.cpp
    src/workload/workload.cpp
    src/report/stats.cpp
    src/report/chart.cpp
    src/bench/runner.cpp
)
target_include_directories(avis_core PUBLIC include)
target_compile_features(avis_core PUBLIC cxx_std_23)
target_link_libraries(avis_core PRIVATE avis_warnings)

find_package(Threads REQUIRED)
target_link_libraries(avis_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# NVML telemetry
# ---------------------------------------------------------------------------
if(AVIS_WITH_NVML)
    find_path(NVML_INCLUDE_DIR nvml.h PATHS /usr/local/cuda/include /usr/include)
    find_library(NVML_LIBRARY nvidia-ml PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64/stubs)
    if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
        add_library(avis_nvml src/telemetry/nvml_device.cpp)
        target_include_directories(avis_nvml PRIVATE ${NVML_INCLUDE_DIR})
        target_link_libraries(avis_nvml PUBLIC avis_core PRIVATE ${NVML_LIBRARY} avis_warnings)
        target_compile_definitions(avis_nvml PUBLIC AVIS_HAS_NVML=1)
    else()
        message(WARNING "NVML not found: building without GPU telemetry")
        set(AVIS_WITH_NVML OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# llama.cpp backend and the avis CLI
# ---------------------------------------------------------------------------
if(AVIS_WITH_LLAMA)
    find_package(llama CONFIG)
    if(llama_FOUND)
        add_library(avis_llama src/llama/llama_executor.cpp)
        target_link_libraries(avis_llama PUBLIC avis_core llama PRIVATE avis_warnings)

        add_executable(avis apps/avis.cpp)
        target_link_libraries(avis PRIVATE avis_core avis_llama avis_warnings)
        if(TARGET avis_nvml)
            target_link_libraries(avis PRIVATE avis_nvml)
        endif()
    else()
        message(WARNING "llama.cpp not found (run scripts/setup_deps.sh): building without the CLI")
        set(AVIS_WITH_LLAMA OFF)
    endif()
endif()

if(AVIS_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(AVIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
