//
// Created by smher on 18-3-5.
//

#include "myslam/frame.h"

namespace myFrontEnd {
    Frame::Frame(): id_(-1), timeStamp_(-1), camera_(nullptr) {
    }

    Frame::Frame(long id, double time, Sophus::SE3 Tcw, Camera::Ptr camera, cv::Mat color, cv::Mat depth) : id_(id),
                                                                                                    timeStamp_(time),
                                                                                                    Tcw_(Tcw),
                                                                                                    camera_(camera),
                                                                                                    color_(color),
                                                                                                    depth_(depth)
    {
    }

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0;
        return Frame::Ptr(new Frame(factory_id));
    }

    double Frame::findDepth(const cv::KeyPoint &kp)
    {
        int x = cvRound(kp.pt.x);    // the columns ! ! !
        int y = cvRound(kp.pt.y);    // the Rows    ! ! !
        ushort dep = depth_.ptr<ushort>(y)[x];
        if (dep != 0)
        {
            return static_cast<double>(dep) / camera_->depth_scale_;
        }
        else    // return the one depth of its four nearest neighbors
        {
            int dx[4] = {-1, 0, 1, 0};
            int dy[4] = {0, -1, 0, 1};

            for (int i = 0; i < 4; ++i)   // rows changing
            {
                for (int j = 0; j < 4; ++j)       // cols changing
                {
                    dep = depth_.ptr<ushort>(y+dy[i])[x+dx[j]];
                    if (dep != 0)
                        return static_cast<double>(dep) / camera_->depth_scale_;
                }
            }
        }

        return -1.0;
        //return depth_.at<double>(kp.pt);   // at function overloaded the Point parameter
    }

    Eigen::Vector3d Frame::getCamCenter() const
    {
        // Tcw_          : the projective transform from world to camera.
        // Tcw_.inverse(): the projective transform from camera to world.
        return Tcw_.inverse().translation();           // TODO: why here have a inverse() ?, 逆过程?
    }

    bool Frame::isInFrame(const Eigen::Vector3d &pt_world)
    {
        Eigen::Vector3d p_c = camera_->world2camera(pt_world, Tcw_);
        if (p_c(3, 0) < 0)
            return false;

        Eigen::Vector2d pixel = camera_->camera2pixel(p_c);
        return pixel(0, 0) > 0 && pixel(0, 0) < color_.cols          // pixel(0, 0): cols ! ! !
                && pixel(1, 0) > 0 && pixel(1, 0) < color_.rows;     // pixel(1, 0): rows ! ! !
    }
}
