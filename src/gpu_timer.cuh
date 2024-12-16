#pragma once

#include <cuda_runtime.h>
#include <source_location>
#include <stdexcept>
#include <string>

namespace easy_timer
{
    class GpuTimer
    {
    protected:
        bool stopped_;
        cudaEvent_t ce_start_;
        cudaEvent_t ce_stop_;

    private:
        static void ThrowOnCudaError(
            const cudaError_t err,
            const std::source_location& location = std::source_location::current()
            )
        {
            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    "CUDA error: " + std::string(cudaGetErrorString(err)) + " at " +
                    location.file_name() + ":" + std::to_string(location.line()) + " " +
                    location.function_name());
            }
        }

    public:
        GpuTimer() : stopped_(true) {
            ThrowOnCudaError(cudaEventCreate(&ce_start_));
            ThrowOnCudaError(cudaEventCreate(&ce_stop_));
        }

        ~GpuTimer()
        {
            if (ce_start_ != nullptr) cudaEventDestroy(ce_start_);
            if (ce_stop_ != nullptr) cudaEventDestroy(ce_stop_);
        }

        void Start()
        {   
            if (!stopped_) return;
            stopped_ = false;
            cudaEventRecord(ce_start_, 0);
        }
        
        void Stop()
        {
            if (stopped_) return;
            stopped_ = true;
            cudaEventRecord(ce_stop_, 0);
            cudaEventSynchronize(ce_stop_);
        }

        void Reset()
        {
            if (!stopped_) this->Stop();
        }
        
        float GetMillis()
        {
            float elapsed_time;
            if (!stopped_) this->Stop();
            cudaEventElapsedTime(&elapsed_time, ce_start_, ce_stop_);
            return elapsed_time;
        }
    };
}
