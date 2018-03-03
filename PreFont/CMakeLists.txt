cmake_minimum_required(VERSION 3.7)
project(SLAMPreFont)

set(CMAKE_CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV REQUIRED)
#message(STATUS "OpenCV Include Path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Others
find_package(Eigen3 REQUIRED)
#message(STATUS "Eigen3 Include Path: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Pangolin REQUIRED)
#message(STATUS "Pangolin Include Path: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(G2O REQUIRED)
#message(STATUS "G2O Include Path: ${G2O_INCLUDE_DIR}")
include_directories(${G2O_INCLUDE_DIR})

# add include directories: my head files
message(STATUS ${PROJECT_SOURCE_DIR})
include_directories({PROJECT_SOURCE_DIR}/include)

# generate share library
add_library(${PROJECT_NAME} SHARED
        src/camera.cpp
)

# add other libraries to the SLAMPreFont.so
set(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear
        g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface
        g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense
        g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
#message(STATUS "G2O Library: ${G2O_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${G2O_LIBS})

# build for executable output
set(SOURCE_FILES main.cpp)

add_executable(mySLAM ${SOURCE_FILES})


# add target link libraries
target_link_libraries(mySLAM ${PROJECT_NAME})
