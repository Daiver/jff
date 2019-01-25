project(FindTBB_pkg)
cmake_minimum_required(VERSION 2.6)

##############################################################################
# Subdirectories
##############################################################################

subdirs(src)


##############################################################################
# Custom find modules
##############################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTBB/")

