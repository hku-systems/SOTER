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
CMAKE_SOURCE_DIR = /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build

# Include any dependencies generated for this target.
include CMakeFiles/gpu_server.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gpu_server.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gpu_server.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gpu_server.dir/flags.make

CMakeFiles/gpu_server.dir/gpu_server.cc.o: CMakeFiles/gpu_server.dir/flags.make
CMakeFiles/gpu_server.dir/gpu_server.cc.o: ../../gpu_server.cc
CMakeFiles/gpu_server.dir/gpu_server.cc.o: CMakeFiles/gpu_server.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gpu_server.dir/gpu_server.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpu_server.dir/gpu_server.cc.o -MF CMakeFiles/gpu_server.dir/gpu_server.cc.o.d -o CMakeFiles/gpu_server.dir/gpu_server.cc.o -c /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/gpu_server.cc

CMakeFiles/gpu_server.dir/gpu_server.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gpu_server.dir/gpu_server.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/gpu_server.cc > CMakeFiles/gpu_server.dir/gpu_server.cc.i

CMakeFiles/gpu_server.dir/gpu_server.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gpu_server.dir/gpu_server.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/gpu_server.cc -o CMakeFiles/gpu_server.dir/gpu_server.cc.s

# Object files for target gpu_server
gpu_server_OBJECTS = \
"CMakeFiles/gpu_server.dir/gpu_server.cc.o"

# External object files for target gpu_server
gpu_server_EXTERNAL_OBJECTS =

gpu_server: CMakeFiles/gpu_server.dir/gpu_server.cc.o
gpu_server: CMakeFiles/gpu_server.dir/build.make
gpu_server: /home/xian/libtorch17/libtorch/lib/libtorch.so
gpu_server: /home/xian/libtorch17/libtorch/lib/libc10.so
gpu_server: /usr/local/cuda-10.1/lib64/stubs/libcuda.so
gpu_server: /usr/local/cuda-10.1/lib64/libnvrtc.so
gpu_server: /usr/local/cuda-10.1/lib64/libnvToolsExt.so
gpu_server: /usr/local/cuda-10.1/lib64/libcudart.so
gpu_server: /home/xian/libtorch17/libtorch/lib/libc10_cuda.so
gpu_server: libhw_grpc_proto.a
gpu_server: /lib/libgrpc++_reflection.a
gpu_server: /lib/libgrpc++.a
gpu_server: /lib/libprotobuf.a
gpu_server: /home/xian/libtorch17/libtorch/lib/libc10_cuda.so
gpu_server: /home/xian/libtorch17/libtorch/lib/libc10.so
gpu_server: /usr/local/cuda-10.1/lib64/libcufft.so
gpu_server: /usr/local/cuda-10.1/lib64/libcurand.so
gpu_server: /usr/lib/x86_64-linux-gnu/libcublas.so
gpu_server: /usr/local/cuda-10.1/lib64/libcudnn.so
gpu_server: /usr/local/cuda-10.1/lib64/libnvToolsExt.so
gpu_server: /usr/local/cuda-10.1/lib64/libcudart.so
gpu_server: /lib/libgrpc.a
gpu_server: /lib/libz.a
gpu_server: /lib/libcares.a
gpu_server: /lib/libaddress_sorting.a
gpu_server: /lib/libre2.a
gpu_server: /lib/libabsl_raw_hash_set.a
gpu_server: /lib/libabsl_hashtablez_sampler.a
gpu_server: /lib/libabsl_hash.a
gpu_server: /lib/libabsl_city.a
gpu_server: /lib/libabsl_low_level_hash.a
gpu_server: /lib/libabsl_statusor.a
gpu_server: /lib/libabsl_bad_variant_access.a
gpu_server: /lib/libgpr.a
gpu_server: /lib/libupb.a
gpu_server: /lib/libabsl_status.a
gpu_server: /lib/libabsl_random_distributions.a
gpu_server: /lib/libabsl_random_seed_sequences.a
gpu_server: /lib/libabsl_random_internal_pool_urbg.a
gpu_server: /lib/libabsl_random_internal_randen.a
gpu_server: /lib/libabsl_random_internal_randen_hwaes.a
gpu_server: /lib/libabsl_random_internal_randen_hwaes_impl.a
gpu_server: /lib/libabsl_random_internal_randen_slow.a
gpu_server: /lib/libabsl_random_internal_platform.a
gpu_server: /lib/libabsl_random_internal_seed_material.a
gpu_server: /lib/libabsl_random_seed_gen_exception.a
gpu_server: /lib/libabsl_cord.a
gpu_server: /lib/libabsl_bad_optional_access.a
gpu_server: /lib/libabsl_cordz_info.a
gpu_server: /lib/libabsl_cord_internal.a
gpu_server: /lib/libabsl_cordz_functions.a
gpu_server: /lib/libabsl_exponential_biased.a
gpu_server: /lib/libabsl_cordz_handle.a
gpu_server: /lib/libabsl_str_format_internal.a
gpu_server: /lib/libabsl_synchronization.a
gpu_server: /lib/libabsl_stacktrace.a
gpu_server: /lib/libabsl_symbolize.a
gpu_server: /lib/libabsl_debugging_internal.a
gpu_server: /lib/libabsl_demangle_internal.a
gpu_server: /lib/libabsl_graphcycles_internal.a
gpu_server: /lib/libabsl_malloc_internal.a
gpu_server: /lib/libabsl_time.a
gpu_server: /lib/libabsl_strings.a
gpu_server: /lib/libabsl_throw_delegate.a
gpu_server: /lib/libabsl_int128.a
gpu_server: /lib/libabsl_strings_internal.a
gpu_server: /lib/libabsl_base.a
gpu_server: /lib/libabsl_spinlock_wait.a
gpu_server: /lib/libabsl_raw_logging_internal.a
gpu_server: /lib/libabsl_log_severity.a
gpu_server: /lib/libabsl_civil_time.a
gpu_server: /lib/libabsl_time_zone.a
gpu_server: /lib/libssl.a
gpu_server: /lib/libcrypto.a
gpu_server: /lib/libprotobuf.a
gpu_server: CMakeFiles/gpu_server.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gpu_server"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu_server.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpu_server.dir/build: gpu_server
.PHONY : CMakeFiles/gpu_server.dir/build

CMakeFiles/gpu_server.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gpu_server.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gpu_server.dir/clean

CMakeFiles/gpu_server.dir/depend:
	cd /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res/cmake/build/CMakeFiles/gpu_server.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gpu_server.dir/depend

