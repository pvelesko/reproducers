# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pvelesko/firstTouch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pvelesko/firstTouch/build

# Include any dependencies generated for this target.
include CMakeFiles/driver.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/driver.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/driver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/driver.dir/flags.make

CMakeFiles/driver.dir/main.cpp.o: CMakeFiles/driver.dir/flags.make
CMakeFiles/driver.dir/main.cpp.o: ../main.cpp
CMakeFiles/driver.dir/main.cpp.o: CMakeFiles/driver.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pvelesko/firstTouch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/driver.dir/main.cpp.o"
	/opt/install/llvm/16.0/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/driver.dir/main.cpp.o -MF CMakeFiles/driver.dir/main.cpp.o.d -o CMakeFiles/driver.dir/main.cpp.o -c /home/pvelesko/firstTouch/main.cpp

CMakeFiles/driver.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/driver.dir/main.cpp.i"
	/opt/install/llvm/16.0/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pvelesko/firstTouch/main.cpp > CMakeFiles/driver.dir/main.cpp.i

CMakeFiles/driver.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/driver.dir/main.cpp.s"
	/opt/install/llvm/16.0/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pvelesko/firstTouch/main.cpp -o CMakeFiles/driver.dir/main.cpp.s

# Object files for target driver
driver_OBJECTS = \
"CMakeFiles/driver.dir/main.cpp.o"

# External object files for target driver
driver_EXTERNAL_OBJECTS =

driver: CMakeFiles/driver.dir/main.cpp.o
driver: CMakeFiles/driver.dir/build.make
driver: /usr/lib/x86_64-linux-gnu/libOpenCL.so
driver: /usr/local/lib/libze_loader.so
driver: CMakeFiles/driver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pvelesko/firstTouch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable driver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/driver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/driver.dir/build: driver
.PHONY : CMakeFiles/driver.dir/build

CMakeFiles/driver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/driver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/driver.dir/clean

CMakeFiles/driver.dir/depend:
	cd /home/pvelesko/firstTouch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pvelesko/firstTouch /home/pvelesko/firstTouch /home/pvelesko/firstTouch/build /home/pvelesko/firstTouch/build /home/pvelesko/firstTouch/build/CMakeFiles/driver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/driver.dir/depend

