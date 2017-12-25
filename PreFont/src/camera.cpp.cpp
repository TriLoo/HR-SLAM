//
// Created by smher on 17-12-25.
//

#include "include/camera.h"

namespace myslam
{
    Camera::Camera()
    {
    }

    Vector3d Camera::world2camera(const Vectro3d& p_w, const SE3& T_c_w)
    {
        return T_c_w * p_w;
    }
}

