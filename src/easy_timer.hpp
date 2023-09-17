#ifndef __EASY_TIMER_HPP
#define __EASY_TIMER_HPP

#include <cuda_runtime.h>
#include <time.h>

namespace easy_timer
{
    class CpuTimer
    {
    private:
        int stopped;
        timespec ts_start;
        timespec ts_end;

    public:
        CpuTimer()
        {
            stopped = 0;
            ts_start.tv_sec = ts_start.tv_nsec = 0;
            ts_end.tv_sec = ts_end.tv_nsec = 0;
        }
        void start()
        {
            stopped = 0;
            clock_gettime(CLOCK_REALTIME, &ts_start);
        }
        void stop()
        {
            stopped = 1;
            clock_gettime(CLOCK_REALTIME, &ts_end);
        }
        void reset()
        {
            stopped = 0;
            ts_start.tv_sec = ts_start.tv_nsec = 0;
            ts_end.tv_sec = ts_end.tv_nsec = 0;
        }
        double getElapsedTime()
        {
            double elapsedTime;
            if (!stopped)
            {
                clock_gettime(CLOCK_REALTIME, &ts_end);
            }
            elapsedTime = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0;
            elapsedTime += (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000.0;
            return elapsedTime;
        }
    };

    class GpuTimer
    {
    private:
        int stopped = 0;
        cudaEvent_t ce_start = nullptr;
        cudaEvent_t ce_stop = nullptr;

    public:
        GpuTimer()
        {
            stopped = 0;
        }
        ~GpuTimer()
        {
            if (&ce_start != nullptr)
            {
                cudaEventDestroy(ce_start);
            };
            if (&ce_stop != nullptr)
            {
                cudaEventDestroy(ce_stop);
            };
        }
        void start()
        {
            this->reset();
            cudaEventRecord(ce_start, 0);
        }
        void stop()
        {
            stopped = 1;
            cudaEventRecord(ce_stop, 0);
            cudaEventSynchronize(ce_stop);
        }
        void reset()
        {
            stopped = 0;
            if (&ce_start != nullptr)
            {
                cudaEventDestroy(ce_start);
            };
            if (&ce_stop != nullptr)
            {
                cudaEventDestroy(ce_stop);
            };
            cudaEventCreate(&ce_start);
            cudaEventCreate(&ce_stop);
        }
        double getElapsedTime()
        {
            float elapsedTime;
            if (!stopped)
            {
                cudaEventRecord(ce_stop, 0);
                cudaEventSynchronize(ce_stop);
            }
            cudaEventElapsedTime(&elapsedTime, ce_start, ce_stop);
            return elapsedTime;
        }
    };
}
#endif