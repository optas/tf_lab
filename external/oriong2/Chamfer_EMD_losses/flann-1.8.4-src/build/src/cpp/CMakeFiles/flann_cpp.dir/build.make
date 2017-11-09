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

# Include any dependencies generated for this target.
include src/cpp/CMakeFiles/flann_cpp.dir/depend.make

# Include the progress variables for this target.
include src/cpp/CMakeFiles/flann_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpp/CMakeFiles/flann_cpp.dir/flags.make

# Object files for target flann_cpp
flann_cpp_OBJECTS =

# External object files for target flann_cpp
flann_cpp_EXTERNAL_OBJECTS =

lib/libflann_cpp.so.1.8.4: src/cpp/CMakeFiles/flann_cpp.dir/build.make
lib/libflann_cpp.so.1.8.4: lib/libflann_cpp_s.a
lib/libflann_cpp.so.1.8.4: src/cpp/CMakeFiles/flann_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../lib/libflann_cpp.so"
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flann_cpp.dir/link.txt --verbose=$(VERBOSE)
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -E cmake_symlink_library ../../lib/libflann_cpp.so.1.8.4 ../../lib/libflann_cpp.so.1.8 ../../lib/libflann_cpp.so

lib/libflann_cpp.so.1.8: lib/libflann_cpp.so.1.8.4

lib/libflann_cpp.so: lib/libflann_cpp.so.1.8.4

# Rule to build all files generated by this target.
src/cpp/CMakeFiles/flann_cpp.dir/build: lib/libflann_cpp.so
.PHONY : src/cpp/CMakeFiles/flann_cpp.dir/build

src/cpp/CMakeFiles/flann_cpp.dir/requires:
.PHONY : src/cpp/CMakeFiles/flann_cpp.dir/requires

src/cpp/CMakeFiles/flann_cpp.dir/clean:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -P CMakeFiles/flann_cpp.dir/cmake_clean.cmake
.PHONY : src/cpp/CMakeFiles/flann_cpp.dir/clean

src/cpp/CMakeFiles/flann_cpp.dir/depend:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/src/cpp /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp/CMakeFiles/flann_cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpp/CMakeFiles/flann_cpp.dir/depend

