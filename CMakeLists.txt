cmake_minimum_required(VERSION 3.16)
project(Avis_Inference)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    llama.cpp/include
    llama.cpp/ggml/include
    /usr/include
)

link_directories(
    llama.cpp/build/bin
    /usr/lib/x86_64-linux-gnu
)

add_executable(avis src/main.cpp)

target_link_libraries(avis
    llama
    ggml
    ggml-base
    ggml-cpu
    ggml-cuda
    nvidia-ml
    pthread
)

set_target_properties(avis PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/llama.cpp/build/bin"
)
