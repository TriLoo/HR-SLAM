add_library(${PROJECT_NAME} SHARED
        PoseGraph.cpp
)

#message(STATUS "${Sophus_LIBRARIES}")
#message(STATUS "${EIGEN3_LIBS}")

set(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear
        g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface
        g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense
        g2o_stuff g2o_types_sclam2d g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse
        )

target_link_libraries( ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${G2O_LIBS}
        ${EIGEN3_LIBS}
)
