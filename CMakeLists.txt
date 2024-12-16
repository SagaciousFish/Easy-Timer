cmake_minimum_required(VERSION 3.30)
project(easy_timer CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)

add_executable(easy_timer_test
    src/cpu_timer.hpp
    src/gpu_timer.cuh
    test/timer_test.cu
)
