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
CMAKE_SOURCE_DIR = /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build

# Utility rule file for tests.

# Include the progress variables for this target.
include test/CMakeFiles/tests.dir/progress.make

test/CMakeFiles/tests:

tests: test/CMakeFiles/tests
tests: test/CMakeFiles/tests.dir/build.make
.PHONY : tests

# Rule to build all files generated by this target.
test/CMakeFiles/tests.dir/build: tests
.PHONY : test/CMakeFiles/tests.dir/build

test/CMakeFiles/tests.dir/clean:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/test && $(CMAKE_COMMAND) -P CMakeFiles/tests.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/tests.dir/clean

test/CMakeFiles/tests.dir/depend:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/test /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/test /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/test/CMakeFiles/tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/tests.dir/depend

