cmake_minimum_required(VERSION 2.8)
project(ANNTest1)

set(CMAKE_CXX_COMPILER "clang++-3.5")

#check c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../FindTBB/CMake/FindTBB/")
find_package(TBB)
if(NOT TBB_FOUND)
    MESSAGE (FATAL_ERROR "TBB not found. Example project will not be built.")
endif()

# Eigen3 Headerfiles
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(../../../CppNumericalSolvers/include/ )

#add_subdirectory(../../../CppNumericalSolvers/include/gtest)

find_package(GTest)

add_executable(main main.cxx)
SET_TARGET_PROPERTIES(main PROPERTIES LINK_FLAGS -Wl,-ladept)
#target_link_libraries (main gtest gtest_main pthread)

#add_subdirectory(CommonEigenRoutine)
#add_subdirectory(CppOpt)
