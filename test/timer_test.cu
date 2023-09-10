#include "../src/easy_timer.hpp"

#include <cuda_runtime.h>
#include <iostream>

__global__ void assign_one(int *a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    a[tid] = 1;
}

void run_with_candidate(int N, int candidate)
{
    easy_timer::GpuTimer timer;
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    int THREADS_PER_BLOCK = candidate;
    int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::cout << "<<<" << NUM_BLOCKS << ", " << THREADS_PER_BLOCK << ">>>" << std::endl;
    timer.start();
    assign_one<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a);
    timer.stop();
    std::cout << timer.getElapsedTime() << std::endl;
    cudaFree(d_a);
}

int main(void)
{
    int N = (256 * 1024 * 1024);
    std::cout << N << std::endl;

    easy_timer::CpuTimer timer;
    int *a;
    a = (int *)malloc(N * sizeof(int));
    timer.start();
    for (int i = 0; i < N; i++)
    {
        a[i] = 1;
    }
    timer.stop();
    free(a);
    std::cout << timer.getElapsedTime() << std::endl;

    int candidates[] = {1, 16, 32, 128, 256, 1024};

    for (int candidate : candidates)
    {
        std::cout << "candidate: " << candidate << std::endl;
        run_with_candidate(N, candidate);
    }
}