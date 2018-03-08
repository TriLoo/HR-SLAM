//
// Created by smher on 18-3-8.
//

#ifndef SLAMPREFONT_MAP_H
#define SLAMPREFONT_MAP_H

#include "system.h"
#include "myslam/MapPoint.h"
#include "myslam/frame.h"

#include "unordered_map"


namespace myFrontEnd
{
    class Map
    {
    public:
        typedef std::shared_ptr<Map> Ptr;
        Map();

        std::unordered_map<unsigned long, MapPoint::Ptr> map_points_;
        std::unordered_map<unsigned long, Frame::Ptr> keyframes_;

        void insertKeyFrame(Frame::Ptr frame);
        void insertMapPoint(MapPoint::Ptr map_point);
    };
}

#endif //SLAMPREFONT_MAP_H
