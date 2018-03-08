//
// Created by smher on 18-3-7.
//

#include "myslam/MapPoint.h"

namespace myFrontEnd
{
    MapPoint::MapPoint():id_(-1) {}
    MapPoint::MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm) : id_(id), pos_(pos), norm_(norm)
    {
    }
    MapPoint::~MapPoint()
    {
    }
}

