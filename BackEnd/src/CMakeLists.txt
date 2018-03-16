add_library(${PROJECT_NAME} SHARED
        PoseGraph.cpp
)

#message(STATUS "${Sophus_LIBRARIES}")
#message(STATUS "${EIGEN3_LIBS}")

set(G2O_LIBS g2o_solver_slam2d_linear)

target_link_libraries( ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${G2O_LIBS}
        ${EIGEN3_LIBS}
)
