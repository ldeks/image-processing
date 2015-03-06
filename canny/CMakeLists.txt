cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Canny)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-g -Wall")

add_executable(Canny Canny.cpp ../LauraConvolution.cpp ../LauraFilters.cpp)
target_link_libraries(Canny ${OpenCV_LIBS})
