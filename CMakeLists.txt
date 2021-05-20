cmake_minimum_required(VERSION 3.7)

project(msg_utils)

MESSAGE("building library msg_utils")

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

FILE(GLOB_RECURSE CPU_SRC "./src/*.cpp" "./src/*.c")
FILE(GLOB_RECURSE GPU_SRC "./src/*.cu")

include_directories(${MPI_INCLUDE_PATH} ./)

add_library(msg_utils STATIC ${CPU_SRC} ${MPI_C_LIBRARIES})
# cuda_add_library(msg_utils_cu STATIC ${GPU_SRC})
