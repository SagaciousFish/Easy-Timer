#pragma once

#include <chrono>

using namespace std::chrono;

#define RAII_TIMER_F(name) easy_timer::CpuTimer timer(name);
#define RAII_TIMER() easy_timer::CpuTimer timer(__FILE__ ":" __LINE__ ":" __func__);

namespace easy_timer {
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


    class CpuTimer {
        protected:
            bool stopped_;
            time_point<TimerClock> tp_start_;
            time_point<TimerClock> tp_end_;
            const char *name_{};

        public:
            CpuTimer()
                : stopped_(true) {
                tp_end_   = TimerClock::now();
                tp_start_ = TimerClock::now();
            }

            explicit CpuTimer(const char *name)
                : CpuTimer() {
                name_ = name;
                this->Start();
            }

            ~CpuTimer() {
                if (!stopped_) {
                    printf("[RAII] %s: %.5f ms\n", name_, this->GetMillis());
                }
            }


            void Start() {
                if (!stopped_)
                    return;
                stopped_  = false;
                tp_start_ = TimerClock::now();
            }

            void Stop() {
                stopped_ = true;
                tp_end_  = TimerClock::now();
            }

            float GetMillis() {
                tp_end_                  = TimerClock::now();
                const float elapsed_time = duration<float, std::milli>(tp_end_ - tp_start_).count();
                return elapsed_time;
            }
    };
}
