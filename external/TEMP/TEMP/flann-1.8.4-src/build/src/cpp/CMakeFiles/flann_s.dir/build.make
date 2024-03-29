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
include src/cpp/CMakeFiles/flann_s.dir/depend.make

# Include the progress variables for this target.
include src/cpp/CMakeFiles/flann_s.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpp/CMakeFiles/flann_s.dir/flags.make

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o: src/cpp/CMakeFiles/flann_s.dir/flags.make
src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o: ../src/cpp/flann/flann.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o"
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/flann_s.dir/flann/flann.cpp.o -c /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/src/cpp/flann/flann.cpp

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flann_s.dir/flann/flann.cpp.i"
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/src/cpp/flann/flann.cpp > CMakeFiles/flann_s.dir/flann/flann.cpp.i

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flann_s.dir/flann/flann.cpp.s"
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/src/cpp/flann/flann.cpp -o CMakeFiles/flann_s.dir/flann/flann.cpp.s

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.requires:
.PHONY : src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.requires

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.provides: src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.requires
	$(MAKE) -f src/cpp/CMakeFiles/flann_s.dir/build.make src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.provides.build
.PHONY : src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.provides

src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.provides.build: src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o

# Object files for target flann_s
flann_s_OBJECTS = \
"CMakeFiles/flann_s.dir/flann/flann.cpp.o"

# External object files for target flann_s
flann_s_EXTERNAL_OBJECTS =

lib/libflann_s.a: src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o
lib/libflann_s.a: src/cpp/CMakeFiles/flann_s.dir/build.make
lib/libflann_s.a: src/cpp/CMakeFiles/flann_s.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../../lib/libflann_s.a"
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -P CMakeFiles/flann_s.dir/cmake_clean_target.cmake
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flann_s.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpp/CMakeFiles/flann_s.dir/build: lib/libflann_s.a
.PHONY : src/cpp/CMakeFiles/flann_s.dir/build

src/cpp/CMakeFiles/flann_s.dir/requires: src/cpp/CMakeFiles/flann_s.dir/flann/flann.cpp.o.requires
.PHONY : src/cpp/CMakeFiles/flann_s.dir/requires

src/cpp/CMakeFiles/flann_s.dir/clean:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp && $(CMAKE_COMMAND) -P CMakeFiles/flann_s.dir/cmake_clean.cmake
.PHONY : src/cpp/CMakeFiles/flann_s.dir/clean

src/cpp/CMakeFiles/flann_s.dir/depend:
	cd /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/src/cpp /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp /orions4-zfs/projects/fhq/cuda/flann-1.8.4-src/build/src/cpp/CMakeFiles/flann_s.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpp/CMakeFiles/flann_s.dir/depend

