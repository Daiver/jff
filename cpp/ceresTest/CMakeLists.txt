cmake_minimum_required(VERSION 2.8)

project(helloworld)

find_package(Ceres REQUIRED PATHS "/home/daiver/coding/ceres-bin/install/")
include_directories(${CERES_INCLUDE_DIRS})

# helloworld
add_executable(regression1 regression1.cpp)
target_link_libraries(regression1 ${CERES_LIBRARIES})

add_executable(tmp tmp.cpp)
target_link_libraries(tmp ${CERES_LIBRARIES})

