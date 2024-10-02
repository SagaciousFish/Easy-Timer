#pragma once

#include <chrono>

using namespace std::chrono;

namespace easy_timer
{
    class CpuTimer
    {
    private:
        bool stopped;
        time_point<steady_clock> tp_start;
        time_point<steady_clock> tp_end;

    public:
        CpuTimer() : stopped(true)
        {
            tp_start = steady_clock::now();
            tp_end = steady_clock::now();
        }
        void start()
        {
            if (!stopped)
                return;
            stopped = false;
            tp_start = steady_clock::now();
        }
        void stop()
        {
            stopped = 1;
            tp_end = steady_clock::now();
        }
        float getMillis()
        {
            float elapsedTime;
            if (!stopped)
                tp_end = steady_clock::now();
            elapsedTime = duration<float, std::milli>(tp_end - tp_start).count();
            return elapsedTime;
        }
    };
}