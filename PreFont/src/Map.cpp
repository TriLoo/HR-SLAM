//
// Created by smher on 18-3-8.
//

#include "myslam/Map.h"

namespace myFrontEnd
{
    //Map::Map(): map_points_(std::make_pair<unsigned long, MapPoint::Ptr>(0, nullptr)), keyframes_(std::make_pair<unsigned long, Frame::Ptr>(0, nullptr))
    Map::Map()
    {}

    void Map::insertKeyFrame(Frame::Ptr frame)
    {
        if (keyframes_.find(frame->id_) == keyframes_.end())
            keyframes_.insert({frame->id_, frame});
        else
            keyframes_[frame->id_] = frame;
    }

    void Map::insertMapPoint(MapPoint::Ptr map_point)
    {
        if (map_points_.find(map_point->id_) == map_points_.end())    // Add new map point
            map_points_.insert(std::make_pair(map_point->id_, map_point));
        else                                                          // Update
            map_points_[map_point->id_] = map_point;
    }
}

