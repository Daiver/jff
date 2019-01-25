project(FindTBB_example)
cmake_minimum_required(VERSION 2.6)

##############################################################################
# Configuration
##############################################################################

# If your TBB install directory is not found automatically, enter it here or use TBB_INSTALL_DIR env variable. (w/o trailing slash)
#set(TBB_INSTALL_DIR "...")
# Enter your architecture [ia32|em64t|itanium] here
set(TBB_ARCHITECTURE "ia32")
# If your compiler is not detected automatically, enter it here. (e.g. vc9 or cc3.2.3_libc2.3.2_kernel2.4.21 or cc4.0.1_os10.4.9)
#set(TBB_COMPILER "...")


##############################################################################
# Find TBB
##############################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../CMake/FindTBB/")
find_package(TBB)
# Or use: find_package(TBB REQUIRED)

if(NOT TBB_FOUND)
        MESSAGE (STATUS "TBB not found. Example project will not be built.")
else(NOT TBB_FOUND)


##############################################################################
# Sources
##############################################################################

set(FindTBB_example_SRCS_H
)

set(FindTBB_example_SRCS_CPP
    sub_string_finder.cpp
)


##############################################################################
# Includes
##############################################################################

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TBB_INCLUDE_DIRS}
)


##############################################################################
# Linking
##############################################################################

link_directories(
    ${TBB_LIBRARY_DIRS}
)


##############################################################################
# Targets
##############################################################################

add_executable(FindTBB_example
    ${FindTBB_example_SRCS_CPP} 
    ${FindTBB_example_SRCS_H}
)

#target_link_libraries(FindTBB_example
#    ${TBB_LIBRARIES}
#)


endif(NOT TBB_FOUND)
