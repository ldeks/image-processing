cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(logEdge)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-g -Wall")

add_executable(logEdge logEdge.cpp ../LauraConvolution.cpp ../LauraFilters.cpp)
target_link_libraries(logEdge ${OpenCV_LIBS})
