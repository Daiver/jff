cmake_minimum_required(VERSION 3.0)
project(Example)

include_directories(/usr/include/eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(SRC main.cpp)

add_executable(main ${SRC})
