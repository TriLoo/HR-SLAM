cmake_minimum_required(VERSION 3.7)
project(SLAMPreFont)

set(CMAKE_CXX_STANDARD 11)

# add include directories: head files
include_directories(./include)

set(SOURCE_FILES main.cpp)

add_executable(SLAMPreFont ${SOURCE_FILES})

