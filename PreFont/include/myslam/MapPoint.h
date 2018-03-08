//
// Created by smher on 18-3-7.
//

#ifndef SLAMPREFONT_MAPPOINT_H
#define SLAMPREFONT_MAPPOINT_H

#include "system.h"

namespace myFrontEnd
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        MapPoint();
        MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm);
        ~MapPoint();

        unsigned long id_;
        Eigen::Vector3d pos_, norm_;

        //std::vector<cv::KeyPoint> kps_;
        cv::Mat descriptor_;
        int observed_times_;
        int matched_times_;

        // factory function
        static Ptr createMapPoint();
    private:

    };
}

#endif //SLAMPREFONT_MAPPOINT_H
