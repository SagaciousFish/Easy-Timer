#pragma once

#include <chrono>

using namespace std::chrono;

namespace easy_timer
{
    // Clock Selection from https://stackoverflow.com/questions/38252022/does-standard-c11-guarantee-that-high-resolution-clock-measure-real-time-non
    // The implementation chooses library-preferred steady clock, or the most precise built-in clock
    using PreciseBuiltinClock =
    std::conditional_t<
        system_clock::period::den <= steady_clock::period::den,
        system_clock, steady_clock
      >;
    using TimerClock =
        std::conditional_t<
            high_resolution_clock::is_steady,
            high_resolution_clock, PreciseBuiltinClock
          >;


    class CpuTimer
    {
    protected:
        bool stopped_;
        time_point<TimerClock> tp_start_;
        time_point<TimerClock> tp_end_;

    public:
        CpuTimer() : stopped_(true)
        {
            tp_start_ = TimerClock::now();
            tp_end_ = TimerClock::now();
        }
        void Start()
        {
            if (!stopped_)
                return;
            stopped_ = false;
            tp_start_ = TimerClock::now();
        }
        void Stop()
        {
            stopped_ = true;
            tp_end_ = TimerClock::now();
        }
        float GetMillis()
        {
            if (!stopped_)
                tp_end_ = TimerClock::now();
            const float elapsed_time = duration<float, std::milli>(tp_end_ - tp_start_).count();
            return elapsed_time;
        }
    };
}
