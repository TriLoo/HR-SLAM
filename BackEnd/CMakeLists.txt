cmake_minimum_required(VERSION 3.7)
project(BackEnd)

set(CMAKE_CXX_STANDARD 11)
include(CheckCXXCompilerFlag)
# Check that the <flag> is accepted by the compiler without a diagnostic. Stores the result in an internal cache entry named <var>.
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
endif ()

# list(APPEND <list> [<element> ...])
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# find OpenCV
find_package(OpenCV REQUIRED)

# find G2O
find_package(G2O REQUIRED)

# find Sophus
find_package(Sophus REQUIRED)

# find Eigen
find_package(Eigen3 REQUIRED)

# add include directories
#message(STATUS "G2O Path: ${G2O_INCLUDE_DIR}")
#message(STATUS "Sophus Path: ${Sophus_INCLUDE_DIR}")
#message(STATUS "Eigen Path: ${EIGEN3_INCLUDE_DIR}")
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# set the output path of executable files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# set the output path of shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)
