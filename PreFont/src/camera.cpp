//
// Created by smher on 17-12-25.
//

//#include "myslam/camera.h"
#include "camera.h"

namespace myFrontEnd
{
    Camera::Camera()
    {
    }

    Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d& p_w, const Sophus::SE3& T_c_w)
    {
        return T_c_w * p_w;
    }
    Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d& p_c, const Sophus::SE3& T_c_w)
    {
        //Eigen::Vector3d p_t = T_c_w.transpose();
        //return
    }
}

