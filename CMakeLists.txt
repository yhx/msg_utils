cmake_minimum_required(VERSION 3.5)

project(msg_utils) 

MESSAGE("building library msg_utils")

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

FILE(GLOB_RECURSE CPU_SRC "./src/*.cpp" "./src/*.c")
FILE(GLOB_RECURSE GPU_SRC "./src/*.cu")
FILE(GLOB GPU_TEST "test/*.cu")

include_directories(${MPI_INCLUDE_PATH} ./include)

add_library(msg_utils STATIC ${CPU_SRC} ${MPI_C_LIBRARIES})
set_target_properties(msg_utils PROPERTIES LINKER_LANGUAGE CXX)
# cuda_add_library(msg_utils_cu STATIC ${GPU_SRC})

FOREACH(exec_path ${GPU_TEST})
	#MESSAGE("EXEC_PATH: ${exec_path}")
	GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
	#MESSAGE("EXEC_FILE: ${exec_file}")
	CUDA_ADD_EXECUTABLE(${exec_file} ${exec_path})
	TARGET_LINK_LIBRARIES(${exec_file} msg_utils nccl ${MPI_LIBRARIES})
ENDFOREACH()
