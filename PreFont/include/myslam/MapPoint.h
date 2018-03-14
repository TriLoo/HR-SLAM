//
// Created by smher on 18-3-7.
//

#ifndef SLAMPREFONT_MAPPOINT_H
#define SLAMPREFONT_MAPPOINT_H

#include "system.h"
#include "frame.h"

namespace myFrontEnd
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        MapPoint();
        MapPoint(
                unsigned long id,
                const Eigen::Vector3d& position,
                const Eigen::Vector3d& norm,
                Frame* frame = nullptr,
                //Frame* frame = nullptr;
                const cv::Mat& descriptor = cv::Mat()
        );
        ~MapPoint();

        unsigned long id_;
        static unsigned long factory_id_;
        bool good_;
        Eigen::Vector3d pos_, norm_;    // 相机的位姿，观测的方向

        //std::vector<cv::KeyPoint> kps_;
        cv::Mat descriptor_;    // 描述子，用于匹配
        std::list<Frame*> observed_frames_;
        //std::list<Frame::Ptr> observed_frames_;

        int observed_times_;    // 被观测的次数
        int matched_times_;     // 被匹配的次数

        inline cv::Point3f getPositionCV() const
        {
            return cv::Point3f(pos_(0, 0), pos_(1, 0), pos_(2, 0));
        }

        // factory function
        static Ptr createMapPoint();
        static Ptr createMapPoint(
                const Eigen::Vector3d& pos_world,
                const Eigen::Vector3d& norm,
                const cv::Mat& descriptor,
                Frame* frame
        );
    private:

    };
    // 定义static成员变量
    // unsigned long MapPoint::factory_id_ = 0;  // error: duplicate defining
}

#endif //SLAMPREFONT_MAPPOINT_H
