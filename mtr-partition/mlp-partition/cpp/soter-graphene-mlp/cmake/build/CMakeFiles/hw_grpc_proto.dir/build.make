# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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
CMAKE_COMMAND = /snap/cmake/1082/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1082/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build

# Include any dependencies generated for this target.
include CMakeFiles/hw_grpc_proto.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hw_grpc_proto.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hw_grpc_proto.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hw_grpc_proto.dir/flags.make

helloworld.pb.cc: /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/protos/helloworld.proto
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating helloworld.pb.cc, helloworld.pb.h, helloworld.grpc.pb.cc, helloworld.grpc.pb.h"
	/bin/protoc-3.19.4.0 --grpc_out /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build --cpp_out /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build -I /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/protos --plugin=protoc-gen-grpc="/bin/grpc_cpp_plugin" /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/protos/helloworld.proto

helloworld.pb.h: helloworld.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate helloworld.pb.h

helloworld.grpc.pb.cc: helloworld.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate helloworld.grpc.pb.cc

helloworld.grpc.pb.h: helloworld.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate helloworld.grpc.pb.h

CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o: CMakeFiles/hw_grpc_proto.dir/flags.make
CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o: helloworld.grpc.pb.cc
CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o: CMakeFiles/hw_grpc_proto.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o -MF CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o.d -o CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o -c /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.grpc.pb.cc

CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.grpc.pb.cc > CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.i

CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.grpc.pb.cc -o CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.s

CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o: CMakeFiles/hw_grpc_proto.dir/flags.make
CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o: helloworld.pb.cc
CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o: CMakeFiles/hw_grpc_proto.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o -MF CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o.d -o CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o -c /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.pb.cc

CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.pb.cc > CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.i

CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/helloworld.pb.cc -o CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.s

# Object files for target hw_grpc_proto
hw_grpc_proto_OBJECTS = \
"CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o" \
"CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o"

# External object files for target hw_grpc_proto
hw_grpc_proto_EXTERNAL_OBJECTS =

libhw_grpc_proto.a: CMakeFiles/hw_grpc_proto.dir/helloworld.grpc.pb.cc.o
libhw_grpc_proto.a: CMakeFiles/hw_grpc_proto.dir/helloworld.pb.cc.o
libhw_grpc_proto.a: CMakeFiles/hw_grpc_proto.dir/build.make
libhw_grpc_proto.a: CMakeFiles/hw_grpc_proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libhw_grpc_proto.a"
	$(CMAKE_COMMAND) -P CMakeFiles/hw_grpc_proto.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hw_grpc_proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hw_grpc_proto.dir/build: libhw_grpc_proto.a
.PHONY : CMakeFiles/hw_grpc_proto.dir/build

CMakeFiles/hw_grpc_proto.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hw_grpc_proto.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hw_grpc_proto.dir/clean

CMakeFiles/hw_grpc_proto.dir/depend: helloworld.grpc.pb.cc
CMakeFiles/hw_grpc_proto.dir/depend: helloworld.grpc.pb.h
CMakeFiles/hw_grpc_proto.dir/depend: helloworld.pb.cc
CMakeFiles/hw_grpc_proto.dir/depend: helloworld.pb.h
	cd /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp/cmake/build/CMakeFiles/hw_grpc_proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hw_grpc_proto.dir/depend

