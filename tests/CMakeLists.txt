include(FetchContent)

FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(avis_tests
    test_block_manager.cpp
    test_scheduler.cpp
    test_power_sampler.cpp
)
target_include_directories(avis_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(avis_tests PRIVATE avis_core avis_warnings GTest::gtest_main)
gtest_discover_tests(avis_tests)
