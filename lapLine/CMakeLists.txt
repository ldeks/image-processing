cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lapLine)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-g -Wall")

add_executable(lapLine lapLine.cpp ../LauraConvolution.cpp ../LauraFilters.cpp)
target_link_libraries(lapLine ${OpenCV_LIBS})
