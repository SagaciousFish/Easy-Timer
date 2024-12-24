# build the test
add_executable(easy_timer_test
        timer_test.cu
)
target_link_libraries(easy_timer_test ${PROJECT_NAME})
