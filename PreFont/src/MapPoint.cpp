//
// Created by smher on 18-3-7.
//

#include "myslam/MapPoint.h"

namespace myFrontEnd
{
    MapPoint::MapPoint():id_(-1), pos_(Eigen::Vector3d(0, 0, 0)), norm_(Eigen::Vector3d(0, 0, 0)), good_(
            true),observed_times_(0), matched_times_(0) {}

    MapPoint::MapPoint(unsigned long id, const Eigen::Vector3d &position, const Eigen::Vector3d &norm, Frame* frame,
                       const cv::Mat &descriptor):id_(-1), pos_(Eigen::Vector3d(0, 0, 0)), norm_(Eigen::Vector3d(0, 0, 0)), good_(
            true),observed_times_(1), matched_times_(1), descriptor_(descriptor)
    {
        observed_frames_.push_back(frame);
    }
    MapPoint::~MapPoint()
    {
    }

    // Factory function
    // No need to have 'static' at the head position
    MapPoint::Ptr MapPoint::createMapPoint()
    {
        return MapPoint::Ptr(new MapPoint(factory_id_++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
    }

    MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d &pos_world, const Eigen::Vector3d &norm,
                                           const cv::Mat &descriptor, Frame* frame)
    {
        return MapPoint::Ptr(new MapPoint(factory_id_++, pos_world, norm, frame, descriptor));
    }

    // 定义static成员变量
    unsigned long MapPoint::factory_id_ = 0;
}

