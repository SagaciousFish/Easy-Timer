#include "../src/gpu_timer.cuh"
#include "../src/cpu_timer.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <memory>

__global__ void assign_one(int *a, const int n)
{
    if (const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n) a[tid] = 1;
}

void run_with_candidate(const int n, const int candidate)
{
    easy_timer::GpuTimer timer;
    int *d_a;
    cudaMalloc(reinterpret_cast<void**>(&d_a), n * sizeof(int));
    const int threads_per_block = candidate;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    std::cout << "<<<" << num_blocks << ", " << threads_per_block << ">>>" << std::endl;
    timer.Start();
    assign_one<<<num_blocks, threads_per_block>>>(d_a, n);
    cudaDeviceSynchronize();
    timer.Stop();
    std::cout << timer.GetMillis() << std::endl;
    cudaFree(d_a);
}

void cpu_timer_usage(const int n)
{
    easy_timer::CpuTimer timer;
    std::unique_ptr<int[]> a(new int[n]);
    timer.Start();
    for (int i = 0; i < n; i++) a[i] = 1;
    timer.Stop();
    std::cout << timer.GetMillis() << std::endl;
}

void gpu_timer_usage(const int n)
{
    for (int candidates[] = {1, 16, 32, 48, 64, 128, 256, 1024}; const int candidate : candidates)
        run_with_candidate(n, candidate);
}

int main()
{
    const int n = (256 * 1024 * 1024);

    cpu_timer_usage(n);
    gpu_timer_usage(n);
}
