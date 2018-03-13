add_library(${PROJECT_NAME} SHARED
        camera.cpp
        frame.cpp
        MapPoint.cpp
        Map.cpp
        Config.cpp
        VisualOdometry.cpp
        g2o_types.cpp
        )

# add other libraries to the SLAMPreFont.so
set(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear
        g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface
        g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense
        g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
#message(STATUS "G2O Library: ${G2O_LIBRARIES}")
#message(STATUS "Sophus Library: ${Sophus_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${G2O_LIBS}
        ${Sophus_LIBRARIES})

