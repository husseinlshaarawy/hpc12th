# CMake generated Testfile for 
# Source directory: /home/runner/work/hpc12th/hpc12th
# Build directory: /home/runner/work/hpc12th/hpc12th/_codeql_build_dir
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(UnitTests "/home/runner/work/hpc12th/hpc12th/_codeql_build_dir/bin/unit_tests")
set_tests_properties(UnitTests PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/hpc12th/hpc12th/CMakeLists.txt;118;add_test;/home/runner/work/hpc12th/hpc12th/CMakeLists.txt;0;")
subdirs("_deps/catch2-build")
