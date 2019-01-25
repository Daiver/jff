cmake_minimum_required(VERSION 2.8)


#SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++")
SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++-3.6")
#SET (CMAKE_CXX_FLAGS "-O3 --std=c++11 ")
#SET (CMAKE_CXX_FLAGS "-O3 --std=c++14 -lpython2.7")
SET (CMAKE_CXX_FLAGS "-O3 --std=c++1z -lpython2.7")

FIND_PACKAGE( Boost 1.50 COMPONENTS program_options REQUIRED )
FIND_PACKAGE(Boost 1.45.0 COMPONENTS python)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
INCLUDE_DIRECTORIES(/usr/include/python2.7/)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR} "/home/daiver/coding/jff/cpp/Linq/")
INCLUDE_DIRECTORIES("/home/daiver/coding/jff/cpp/hana/include/")
INCLUDE_DIRECTORIES("/home/daiver/coding/jff/cpp/cpplinq/")

#find_package(BOOST)

add_library(pblib SHARED pblib.cc)
target_link_libraries(pblib ${Boost_LIBRARIES})

add_executable(main main.cc)
target_link_libraries(main ${Boost_LIBRARIES})

add_executable(cpplinqtest cpplinqtest.cc)
target_link_libraries(cpplinqtest ${Boost_LIBRARIES})

add_executable(fusionExamples fusionExamples.cc)
target_link_libraries(fusionExamples ${Boost_LIBRARIES})

add_executable(fusionExamples2 fusionExamples2.cc)
target_link_libraries(fusionExamples2 ${Boost_LIBRARIES})

add_executable(fusionExamples3 fusionExamples3.cc)
target_link_libraries(fusionExamples3 ${Boost_LIBRARIES})

add_executable(fusionExamples4 fusionExamples4.cc)
target_link_libraries(fusionExamples4 ${Boost_LIBRARIES})

add_executable(proto proto.cpp)
target_link_libraries(proto ${Boost_LIBRARIES})

add_executable(proto2 proto2.cpp)
target_link_libraries(proto2 ${Boost_LIBRARIES})

add_executable(proto3 proto3.cpp)
target_link_libraries(proto3 ${Boost_LIBRARIES})

add_executable(phoenix phoenix.cpp)
target_link_libraries(phoenix ${Boost_LIBRARIES})

add_executable(linqtest linqtest.cc)
target_link_libraries(linqtest ${Boost_LIBRARIES})

add_executable(hana hana.cpp)
#target_link_libraries(linqtest ${Boost_LIBRARIES})

#add_executable(hanaTest hanaTest.cc)
#target_link_libraries(hanaTest ${Boost_LIBRARIES})

