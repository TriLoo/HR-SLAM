//
// Created by smher on 17-12-24.
//

#ifndef SLAMPREFONT_CAMERA_H
#define SLAMPREFONT_CAMERA_H

#include "memory"

#include "Eigen/Core"
#include "sophus/se3.h"
#include "sophus/so3.h"

namespace myFrontEnd
{
    class Camera
    {
    public:
        typedef std::shared_ptr<Camera> Ptr;

        float fx_, fy_, cz_, cy_, depth_scale_;

        Camera();
        Camera(float fx, float fy, float cz, float cy, float depth_scale = 0):fx_(fx), fy_(fy), cz_(cz), cy_(cy), depth_scale_(depth_scale){}


        Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3 & T_c_w);
        Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3 & T_c_w);
        Eigen::Vector3d camera2pixel(const Eigen::Vector3d &p_c);
        Eigen::Vector3d pixel2camera(const Eigen::Vector3d &p_p, double depth=1);
        Eigen::Vector3d pixel2world(const Eigen::Vector3d &p_p, const Sophus::SE3 & T_c_w, double depth = 1);
        Eigen::Vector3d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3 & T_c_w);
    private:
    };
}

#endif //SLAMPREFONT_CAMERA_H
