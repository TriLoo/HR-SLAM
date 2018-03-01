//
// Created by smher on 17-12-24.
//

#ifndef SLAMPREFONT_CAMERA_H
#define SLAMPREFONT_CAMERA_H

namespace myslam
{
    class Camera
    {
    public:
        typedef std::shared_ptr<Camera> Ptr;

        float fx_, fy_, cz_, cy_, depth_scale_;

        Camera();
        Camera(float fx, float fy, float cx, float cy, float depth_scale = 0):fx_(fx), fy_(fy), cz_(cz), cy_(cy), depth_scale_(depth_scale){}


        Vector3d world2camera(const Vector3d &p_w, const SE3 & T_c_w);
        Vector3d camera2world(const Vector3d &p_c, const SE3 & T_c_w);
        Vector3d camera2pixel(const Vector3d &p_c);
        Vector3d pixel2camera(const Vector3d &p_p, double depth=1);
        Vector3d pixel2world(const Vector3d &p_p, const SE3 & T_c_w, double depth = 1);
        Vector3d world2pixel(const Vector3d &p_w, const SE3 & T_c_w);
    private:
    };
}

#endif //SLAMPREFONT_CAMERA_H
