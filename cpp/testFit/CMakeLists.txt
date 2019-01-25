cmake_minimum_required(VERSION 2.8)


#SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS "-O3 --std=c++1y ")
#SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-sign-compare -O3 --std=c++11 ")
#SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-sign-compare  -O3 --std=c++1y  -Wunused-parameter -Wstrict-overflow=5 -Wundef -Wshadow -Wcast-qual -Wcast-align -Wmissing-declarations -Wredundant-decls -Wdisabled-optimization -Woverloaded-virtual -Wctor-dtor-privacy -Wold-style-cast")

    #add_executable(bin/discretize_test discretize_test.cpp discretize.cpp common.cpp)
    include_directories(../Fit/)
    add_executable(bin/main main.cc
        #HornSchunck.cpp
    )

    #target_link_libraries(bin/main ${OpenCV_LIBS} tbb)
    #target_link_libraries(bin/test2 ${TBB_LIBS})
    #link_directories(${TBB_LIBRARY_DIRS})
    #add_executable(FindTBB_example ${FindTBB_example_SRCS_CPP})


