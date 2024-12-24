cmake_minimum_required(VERSION 3.30)
project(easy_timer
        VERSION 0.1.0
        LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# build the library
add_library(${PROJECT_NAME}
        src/cpu_timer.hpp
        src/gpu_timer.cuh
)

# set linker language
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# add the test subdirectory
add_subdirectory(test)

include_directories(src)
