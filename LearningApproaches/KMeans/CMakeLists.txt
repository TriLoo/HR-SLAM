cmake_minimum_required(VERSION 3.7)
project(KMeans)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp KMeans.cpp)
add_executable(KMeans ${SOURCE_FILES})

target_link_libraries(KMeans ${OpenCV_LIBS})
