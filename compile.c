#include<stdio.h>
#include<stdlib.h>

int main()
{
	system("g++ -std=c++17 src/main.cpp -Illama.cpp/include -Illama.cpp/ggml/include -Lllama.cpp/build/bin -L/usr/lib/x86_64-linux-gnu -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lnvidia-ml -Wl,-rpath,$PWD/llama.cpp/build/bin -o avis");
	return 0;
}

