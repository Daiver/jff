# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daiver/coding/jff/cpp/testFit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daiver/coding/jff/cpp/testFit

# Include any dependencies generated for this target.
include CMakeFiles/bin/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bin/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bin/main.dir/flags.make

CMakeFiles/bin/main.dir/main.cc.o: CMakeFiles/bin/main.dir/flags.make
CMakeFiles/bin/main.dir/main.cc.o: main.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/daiver/coding/jff/cpp/testFit/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/bin/main.dir/main.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/bin/main.dir/main.cc.o -c /home/daiver/coding/jff/cpp/testFit/main.cc

CMakeFiles/bin/main.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bin/main.dir/main.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/daiver/coding/jff/cpp/testFit/main.cc > CMakeFiles/bin/main.dir/main.cc.i

CMakeFiles/bin/main.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bin/main.dir/main.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/daiver/coding/jff/cpp/testFit/main.cc -o CMakeFiles/bin/main.dir/main.cc.s

CMakeFiles/bin/main.dir/main.cc.o.requires:
.PHONY : CMakeFiles/bin/main.dir/main.cc.o.requires

CMakeFiles/bin/main.dir/main.cc.o.provides: CMakeFiles/bin/main.dir/main.cc.o.requires
	$(MAKE) -f CMakeFiles/bin/main.dir/build.make CMakeFiles/bin/main.dir/main.cc.o.provides.build
.PHONY : CMakeFiles/bin/main.dir/main.cc.o.provides

CMakeFiles/bin/main.dir/main.cc.o.provides.build: CMakeFiles/bin/main.dir/main.cc.o

# Object files for target bin/main
bin/main_OBJECTS = \
"CMakeFiles/bin/main.dir/main.cc.o"

# External object files for target bin/main
bin/main_EXTERNAL_OBJECTS =

bin/main: CMakeFiles/bin/main.dir/main.cc.o
bin/main: CMakeFiles/bin/main.dir/build.make
bin/main: CMakeFiles/bin/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bin/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bin/main.dir/build: bin/main
.PHONY : CMakeFiles/bin/main.dir/build

CMakeFiles/bin/main.dir/requires: CMakeFiles/bin/main.dir/main.cc.o.requires
.PHONY : CMakeFiles/bin/main.dir/requires

CMakeFiles/bin/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bin/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bin/main.dir/clean

CMakeFiles/bin/main.dir/depend:
	cd /home/daiver/coding/jff/cpp/testFit && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daiver/coding/jff/cpp/testFit /home/daiver/coding/jff/cpp/testFit /home/daiver/coding/jff/cpp/testFit /home/daiver/coding/jff/cpp/testFit /home/daiver/coding/jff/cpp/testFit/CMakeFiles/bin/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bin/main.dir/depend

