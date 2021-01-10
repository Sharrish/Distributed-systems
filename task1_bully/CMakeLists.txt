cmake_minimum_required(VERSION 3.6)
project(task1_bully C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")

include_directories(/usr/include/mpi/)

add_executable(task1_bully bully.c)