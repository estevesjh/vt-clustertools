# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_COMMAND = /tawala/packages/cmake-2.6.4-1/bin/cmake

# The command to remove a file.
RM = /tawala/packages/cmake-2.6.4-1/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fsobreira/athena/athena_1.7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fsobreira/athena/athena_1.7/build

# Include any dependencies generated for this target.
include src/CMakeFiles/venice.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/venice.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/venice.dir/flags.make

src/CMakeFiles/venice.dir/venice.c.o: src/CMakeFiles/venice.dir/flags.make
src/CMakeFiles/venice.dir/venice.c.o: ../src/venice.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fsobreira/athena/athena_1.7/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object src/CMakeFiles/venice.dir/venice.c.o"
	cd /home/fsobreira/athena/athena_1.7/build/src && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/venice.dir/venice.c.o   -c /home/fsobreira/athena/athena_1.7/src/venice.c

src/CMakeFiles/venice.dir/venice.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/venice.dir/venice.c.i"
	cd /home/fsobreira/athena/athena_1.7/build/src && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -E /home/fsobreira/athena/athena_1.7/src/venice.c > CMakeFiles/venice.dir/venice.c.i

src/CMakeFiles/venice.dir/venice.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/venice.dir/venice.c.s"
	cd /home/fsobreira/athena/athena_1.7/build/src && /usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -S /home/fsobreira/athena/athena_1.7/src/venice.c -o CMakeFiles/venice.dir/venice.c.s

src/CMakeFiles/venice.dir/venice.c.o.requires:
.PHONY : src/CMakeFiles/venice.dir/venice.c.o.requires

src/CMakeFiles/venice.dir/venice.c.o.provides: src/CMakeFiles/venice.dir/venice.c.o.requires
	$(MAKE) -f src/CMakeFiles/venice.dir/build.make src/CMakeFiles/venice.dir/venice.c.o.provides.build
.PHONY : src/CMakeFiles/venice.dir/venice.c.o.provides

src/CMakeFiles/venice.dir/venice.c.o.provides.build: src/CMakeFiles/venice.dir/venice.c.o
.PHONY : src/CMakeFiles/venice.dir/venice.c.o.provides.build

# Object files for target venice
venice_OBJECTS = \
"CMakeFiles/venice.dir/venice.c.o"

# External object files for target venice
venice_EXTERNAL_OBJECTS =

src/venice: src/CMakeFiles/venice.dir/venice.c.o
src/venice: src/CMakeFiles/venice.dir/build.make
src/venice: src/CMakeFiles/venice.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable venice"
	cd /home/fsobreira/athena/athena_1.7/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/venice.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/venice.dir/build: src/venice
.PHONY : src/CMakeFiles/venice.dir/build

src/CMakeFiles/venice.dir/requires: src/CMakeFiles/venice.dir/venice.c.o.requires
.PHONY : src/CMakeFiles/venice.dir/requires

src/CMakeFiles/venice.dir/clean:
	cd /home/fsobreira/athena/athena_1.7/build/src && $(CMAKE_COMMAND) -P CMakeFiles/venice.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/venice.dir/clean

src/CMakeFiles/venice.dir/depend:
	cd /home/fsobreira/athena/athena_1.7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fsobreira/athena/athena_1.7 /home/fsobreira/athena/athena_1.7/src /home/fsobreira/athena/athena_1.7/build /home/fsobreira/athena/athena_1.7/build/src /home/fsobreira/athena/athena_1.7/build/src/CMakeFiles/venice.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/venice.dir/depend

