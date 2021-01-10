cmake_minimum_required(VERSION 3.17)
project(task2_fault_tolerance C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")

include_directories(/usr/include/mpi/)

add_executable(task2_fault_tolerance heat3d_basic.c heat3d_MPI.c heat3d_MPI_FT.c)