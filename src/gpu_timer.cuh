#pragma once

#include <cuda_runtime.h>

namespace easy_timer
{
    class GpuTimer
    {
    private:
        bool stopped;
        cudaEvent_t ce_start;
        cudaEvent_t ce_stop;

    public:
        GpuTimer() : stopped(true) {
            cudaEventCreate(&ce_start);
            cudaEventCreate(&ce_stop);
        }
        ~GpuTimer()
        {
            if (&ce_start != nullptr) cudaEventDestroy(ce_start);
            if (&ce_stop != nullptr) cudaEventDestroy(ce_stop);
        }
        void start()
        {   
            if (!stopped) return;
            stopped = false;
            cudaEventRecord(ce_start, 0);
        }
        void stop()
        {
            if (stopped) return;
            stopped = true;
            cudaEventRecord(ce_stop, 0);
            cudaEventSynchronize(ce_stop);
        }
        float getMillis()
        {
            float elapsedTime;
            if (!stopped) this->stop();
            cudaEventElapsedTime(&elapsedTime, ce_start, ce_stop);
            return elapsedTime;
        }
    };
}