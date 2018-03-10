//
// Created by smher on 18-3-5.
//

#ifndef SLAMPREFONT_FRAME_H
#define SLAMPREFONT_FRAME_H

#include "system.h"
#include "camera.h"

namespace myFrontEnd
{
    class Frame
    {
    public:
        typedef std::shared_ptr<Frame> Ptr;

        Frame();
        Frame(long id, double time = 0, Sophus::SE3 Tcw = Sophus::SE3(), Camera::Ptr camera = nullptr,
              cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());
        ~Frame();

        // Factory Mode
        // Singleton mode
        static Ptr createFrame();     // In the class declaration, no need to use Frame::Ptr

        // Find the depth in depth map
        double findDepth(const cv::KeyPoint& kp);

        // Get Camera Center
        Eigen::Vector3d getCamCenter() const;

        // Check if a point is in this frame
        bool isInFrame(const Eigen::Vector3d& pt_world);

    public:
        unsigned long id_;
        double timeStamp_;
        Sophus::SE3 Tcw_;    // The projective transform from world to camera, using Lie Group
        Camera::Ptr camera_;    // To get the camera intrinsic, 内参
        cv::Mat color_, depth_;   // RGBD images

    private:

    };
}


#endif //SLAMPREFONT_FRAME_H
