# Check requirements
cmake_minimum_required(VERSION 3.24)

# Name the project you can choose any name you want here
project(lap)

find_package(OpenMP)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-qopenmp-link=static;-ipo;-marchcore-avx-i>")
endif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-flto;-march=native;-mfpmath=sse>")
endif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_GENERATOR_PLATFORM x64)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Wall;/O2>")
endif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

# Define where your executables should be put
set(EXECUTABLE_OUTPUT_PATH .)

# Tell your source files here
set(test_cpu_SRCS
	  ../test/test_cpu.cpp
)

# Include directories
include_directories(../)

add_executable(test_cpu ${test_cpu_SRCS})
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_cpu PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

# second build for counting evaluations
add_executable(test_cpu_evaluated ${test_cpu_SRCS})
target_compile_definitions(test_cpu_evaluated PUBLIC LAP_DISPLAY_EVALUATED)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_cpu_evaluated PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

set (passRegex "test passed")
set (failRegex "test failed")

add_test(NAME sanity_sequential_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon)
add_test(NAME sanity_sequential_epsilon_4k COMMAND test_cpu -table_min 4000 -table_max 4000 -sanity -double -epsilon)
add_test(NAME sanity_sequential_caching_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon -caching)
add_test(NAME sanity_sequential_caching_epsilon_4k COMMAND test_cpu -table_min 4000 -table_max 4000 -sanity -double -epsilon -caching)
if (OpenMP_CXX_FOUND)
  add_test(NAME sanity_parallel_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon -omp)
  add_test(NAME sanity_parallel_epsilon_4k COMMAND test_cpu -table_min 4000 -table_max 4000 -sanity -double -epsilon -omp)
  add_test(NAME sanity_parallel_caching_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon -caching -omp)
  add_test(NAME sanity_parallel_caching_epsilon_4k COMMAND test_cpu -table_min 4000 -table_max 4000 -sanity -double -epsilon -caching -omp)
  add_test(NAME sanity_parallel_caching_epsilon_16k COMMAND test_cpu -table_min 16000 -table_max 16000 -sanity -double -epsilon -caching -omp)
endif (OpenMP_CXX_FOUND)

set_property(TEST sanity_sequential_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_epsilon_4k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_epsilon_4k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_caching_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_caching_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_caching_epsilon_4k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_caching_epsilon_4k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
if (OpenMP_CXX_FOUND)
  set_property(TEST sanity_parallel_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_epsilon_4k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_epsilon_4k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_4k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_4k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_16k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_caching_epsilon_16k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
endif (OpenMP_CXX_FOUND)
