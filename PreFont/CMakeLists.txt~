cmake_minimum_required(VERSION 3.7)
project(SLAMPreFont)
#message(STATUS ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# add include directories: my head files
#message(STATUS ${PROJECT_SOURCE_DIR}/include)
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        #${PROJECT_SOURCE_DIR}/include/myslam
)

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
#message(STATUS "OpenCV Include Path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Others
find_package(Eigen3 REQUIRED)
#message(STATUS "Eigen3 Include Path: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
#message(STATUS ${Sophus_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})

# add Pangolin
#find_package(Pangolin REQUIRED)
#message(STATUS "Pangolin Include Path: ${Pangolin_INCLUDE_DIRS}")
#include_directories(${Eigen3_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
#message(STATUS "G2O Include Path: ${G2O_INCLUDE_DIR}")
include_directories(${G2O_INCLUDE_DIR})

# generate share library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# build for executable output
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


add_subdirectory(src)
add_subdirectory(test)
