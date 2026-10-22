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
    test_clock_policy.cpp
    test_engine.cpp
    test_workload_report.cpp
)
target_include_directories(avis_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(avis_tests PRIVATE AVIS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(avis_tests PRIVATE avis_core avis_warnings GTest::gtest_main)
gtest_discover_tests(avis_tests)

if(TARGET avis_llama)
    add_executable(avis_integration_tests test_llama_integration.cpp)
    target_link_libraries(avis_integration_tests PRIVATE avis_core avis_llama avis_warnings GTest::gtest_main)
    target_compile_definitions(avis_integration_tests PRIVATE AVIS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(avis_integration_tests DISCOVERY_TIMEOUT 60 PROPERTIES LABELS integration)
endif()
