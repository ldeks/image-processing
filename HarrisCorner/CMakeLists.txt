cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(HarrisCorner)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-g -Wall")

add_executable(HarrisCorner HarrisCorner.cpp ../LauraConvolution.cpp ../LauraFilters.cpp)
target_link_libraries(HarrisCorner ${OpenCV_LIBS})
