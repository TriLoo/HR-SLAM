cmake_minimum_required(VERSION 3.7)
project(PCA)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ../cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp PCA.cpp)
add_executable(PCA ${SOURCE_FILES})

target_link_libraries(PCA ${OpenCV_LIBS})
