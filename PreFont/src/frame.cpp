//
// Created by smher on 18-3-5.
//

#include "myslam/frame.h"

namespace myFrontEnd {
    Frame::Frame() {
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
        return std::make_shared();
    }

    double Frame::findDepth(const cv::KeyPoint &kp)
    {
        return depth_.at<double>(kp.pt);   // at function overloaded the Point parameter
    }

    Eigen::Vector3d Frame::getCamCenter() const
    {
        return Eigen::Vector3d(camera_->cx_, camera_->cy_, 1);
    }

    bool Frame::isInFrame(const Eigen::Vector3d &pt_world)
    {
    }
}
