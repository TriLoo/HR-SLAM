//
// Created by smher on 17-12-25.
//

//#include "camera.h"
#include "myslam/camera.h"

namespace myFrontEnd
{
    Camera::Camera()
    {
    }

    // World & Camera : Pc = T * Pw
    Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d& p_w, const Sophus::SE3& T_c_w)
    {
        return T_c_w * p_w;
    }
    Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d& p_c, const Sophus::SE3& T_c_w)
    {
        return T_c_w.inverse() * p_c;
    }

    // Camera & Pixel : Pp = (K * Pc) / depth
    Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
    {
        double depth = p_c(2, 0);
        return Eigen::Vector2d(fx_ * p_c(0, 0) / depth + cx_, fy_ * p_c(1, 0) / depth + cy_);
    }
    Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
    {
        return Eigen::Vector3d(
                (p_p(0, 0) - cx_) * depth / fx_,
                (p_p(1, 0) - cy_) * depth / fy_,
                depth
        );
    }

    // World & Pixel : Pp = (K * T * Pw) / depth
    Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3 &T_c_w)
    {
        return camera2pixel(world2camera(p_w, T_c_w));
    }
    Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3 &T_c_w, double depth)
    {
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }
}

