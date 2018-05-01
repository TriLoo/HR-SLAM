cmake_minimum_required(VERSION 3.7)
project(FeatureExtract)

set(CMAKE_CXX_STANDARD 11)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORT_CXX11)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=C++11)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
#message(STATUS ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost COMPONENTS system chrono thread)
include_directories(${Boost_INCLUDE_DIRS})
#message(STATUS ${Boost_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(FeatureExtract ${SOURCE_FILES})

target_link_libraries(FeatureExtract ${OpenCV_LIBS})
target_link_libraries(FeatureExtract ${Boost_LIBRARIES})
message(STATUS ${EIGEN3_LIBS})
target_link_libraries(FeatureExtract ${EIGEN3_LIBS})
