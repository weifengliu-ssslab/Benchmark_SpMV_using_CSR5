cmake_minimum_required(VERSION 3.16)
project(spmv CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER c++)
#set(CMAKE_CXX_COMPILER icc)
#set(CMAKE_C_COMPILER icc)
#set(CMAKE_C_STANDARD 11)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} ")



add_compile_options(-mavx -mavx2 -march=native -O3)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

link_libraries(m)

add_executable(spmv main.cpp)
