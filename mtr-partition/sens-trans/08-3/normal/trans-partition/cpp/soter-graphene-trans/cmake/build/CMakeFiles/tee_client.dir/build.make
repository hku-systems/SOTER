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
CMAKE_SOURCE_DIR = /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build

# Include any dependencies generated for this target.
include CMakeFiles/tee_client.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tee_client.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tee_client.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tee_client.dir/flags.make

CMakeFiles/tee_client.dir/tee_client.cc.o: CMakeFiles/tee_client.dir/flags.make
CMakeFiles/tee_client.dir/tee_client.cc.o: ../../tee_client.cc
CMakeFiles/tee_client.dir/tee_client.cc.o: CMakeFiles/tee_client.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tee_client.dir/tee_client.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tee_client.dir/tee_client.cc.o -MF CMakeFiles/tee_client.dir/tee_client.cc.o.d -o CMakeFiles/tee_client.dir/tee_client.cc.o -c /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/tee_client.cc

CMakeFiles/tee_client.dir/tee_client.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tee_client.dir/tee_client.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/tee_client.cc > CMakeFiles/tee_client.dir/tee_client.cc.i

CMakeFiles/tee_client.dir/tee_client.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tee_client.dir/tee_client.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/tee_client.cc -o CMakeFiles/tee_client.dir/tee_client.cc.s

# Object files for target tee_client
tee_client_OBJECTS = \
"CMakeFiles/tee_client.dir/tee_client.cc.o"

# External object files for target tee_client
tee_client_EXTERNAL_OBJECTS =

tee_client: CMakeFiles/tee_client.dir/tee_client.cc.o
tee_client: CMakeFiles/tee_client.dir/build.make
tee_client: /home/xian/libtorch17/libtorch/lib/libtorch.so
tee_client: /home/xian/libtorch17/libtorch/lib/libc10.so
tee_client: /usr/local/cuda-10.1/lib64/stubs/libcuda.so
tee_client: /usr/local/cuda-10.1/lib64/libnvrtc.so
tee_client: /usr/local/cuda-10.1/lib64/libnvToolsExt.so
tee_client: /usr/local/cuda-10.1/lib64/libcudart.so
tee_client: /home/xian/libtorch17/libtorch/lib/libc10_cuda.so
tee_client: libhw_grpc_proto.a
tee_client: /lib/libgrpc++_reflection.a
tee_client: /lib/libgrpc++.a
tee_client: /lib/libprotobuf.a
tee_client: /home/xian/libtorch17/libtorch/lib/libc10_cuda.so
tee_client: /home/xian/libtorch17/libtorch/lib/libc10.so
tee_client: /usr/local/cuda-10.1/lib64/libcufft.so
tee_client: /usr/local/cuda-10.1/lib64/libcurand.so
tee_client: /usr/lib/x86_64-linux-gnu/libcublas.so
tee_client: /usr/local/cuda-10.1/lib64/libcudnn.so
tee_client: /usr/local/cuda-10.1/lib64/libnvToolsExt.so
tee_client: /usr/local/cuda-10.1/lib64/libcudart.so
tee_client: /lib/libgrpc.a
tee_client: /lib/libz.a
tee_client: /lib/libcares.a
tee_client: /lib/libaddress_sorting.a
tee_client: /lib/libre2.a
tee_client: /lib/libabsl_raw_hash_set.a
tee_client: /lib/libabsl_hashtablez_sampler.a
tee_client: /lib/libabsl_hash.a
tee_client: /lib/libabsl_city.a
tee_client: /lib/libabsl_low_level_hash.a
tee_client: /lib/libabsl_statusor.a
tee_client: /lib/libabsl_bad_variant_access.a
tee_client: /lib/libgpr.a
tee_client: /lib/libupb.a
tee_client: /lib/libabsl_status.a
tee_client: /lib/libabsl_random_distributions.a
tee_client: /lib/libabsl_random_seed_sequences.a
tee_client: /lib/libabsl_random_internal_pool_urbg.a
tee_client: /lib/libabsl_random_internal_randen.a
tee_client: /lib/libabsl_random_internal_randen_hwaes.a
tee_client: /lib/libabsl_random_internal_randen_hwaes_impl.a
tee_client: /lib/libabsl_random_internal_randen_slow.a
tee_client: /lib/libabsl_random_internal_platform.a
tee_client: /lib/libabsl_random_internal_seed_material.a
tee_client: /lib/libabsl_random_seed_gen_exception.a
tee_client: /lib/libabsl_cord.a
tee_client: /lib/libabsl_bad_optional_access.a
tee_client: /lib/libabsl_cordz_info.a
tee_client: /lib/libabsl_cord_internal.a
tee_client: /lib/libabsl_cordz_functions.a
tee_client: /lib/libabsl_exponential_biased.a
tee_client: /lib/libabsl_cordz_handle.a
tee_client: /lib/libabsl_str_format_internal.a
tee_client: /lib/libabsl_synchronization.a
tee_client: /lib/libabsl_stacktrace.a
tee_client: /lib/libabsl_symbolize.a
tee_client: /lib/libabsl_debugging_internal.a
tee_client: /lib/libabsl_demangle_internal.a
tee_client: /lib/libabsl_graphcycles_internal.a
tee_client: /lib/libabsl_malloc_internal.a
tee_client: /lib/libabsl_time.a
tee_client: /lib/libabsl_strings.a
tee_client: /lib/libabsl_throw_delegate.a
tee_client: /lib/libabsl_int128.a
tee_client: /lib/libabsl_strings_internal.a
tee_client: /lib/libabsl_base.a
tee_client: /lib/libabsl_spinlock_wait.a
tee_client: /lib/libabsl_raw_logging_internal.a
tee_client: /lib/libabsl_log_severity.a
tee_client: /lib/libabsl_civil_time.a
tee_client: /lib/libabsl_time_zone.a
tee_client: /lib/libssl.a
tee_client: /lib/libcrypto.a
tee_client: /lib/libprotobuf.a
tee_client: CMakeFiles/tee_client.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tee_client"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tee_client.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tee_client.dir/build: tee_client
.PHONY : CMakeFiles/tee_client.dir/build

CMakeFiles/tee_client.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tee_client.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tee_client.dir/clean

CMakeFiles/tee_client.dir/depend:
	cd /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build /home/xian/atc22-artifact/backup/soter-graphene/senstrans/08-3/normal/trans-partition/cpp/soter-graphene-trans/cmake/build/CMakeFiles/tee_client.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tee_client.dir/depend

