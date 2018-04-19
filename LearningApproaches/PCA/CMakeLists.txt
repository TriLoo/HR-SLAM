cmake_minimum_required(VERSION 3.7)
project(PCA)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})

set(SOURCE_FILES main.cpp PCA.cpp)
add_executable(PCA ${SOURCE_FILES})

target_link_libraries(PCA ${OpenCV_LIBS} ${EIGEN3_LIBS})
