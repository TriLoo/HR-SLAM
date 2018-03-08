//
// Created by smher on 18-3-7.
//

#include "myslam/MapPoint.h"

namespace myFrontEnd
{
    MapPoint::MapPoint():id_(-1), pos_(Eigen::Vector3d(0, 0, 0)), norm_(Eigen::Vector3d(0, 0, 0)), observed_times_(0), matched_times_(0) {}
    MapPoint::MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm) : id_(id), pos_(pos), norm_(norm), observed_times_(0), matched_times_(0)
    {
    }
    MapPoint::~MapPoint()
    {
    }

    // Factory function
    // No need to have 'static' at the head position
    MapPoint::Ptr MapPoint::createMapPoint()
    {
        static long factory_id = 0;
        return MapPoint::Ptr(new MapPoint(factory_id++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
    }
}

