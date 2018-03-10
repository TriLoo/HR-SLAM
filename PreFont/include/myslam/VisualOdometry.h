//
// Created by smher on 18-3-10.
//

#ifndef SLAMPREFONT_VISUALODOMETRY_H
#define SLAMPREFONT_VISUALODOMETRY_H

#include "system.h"
#include "myslam/Map.h"

namespace myFrontEnd
{
    class VisualOdometry
    {
    public:
        typedef std::shared_ptr<VisualOdometry> Ptr;

        // 不限制范围的枚举
        // 限制范围的枚举: enum class VOState, 必须要加上class关键字
        enum VOState  : int        // use ' : int ' 指定VOState成员的类型，无范围的enum没有默认成员大小(类型)
        {
            INITIALIZING = -1,      // is const variable
            OK = 0,
            LOST
        };

        VOState  state_;
        Map::Ptr map_;             // 包含地图点MapPoint的地图Map
        Frame::Ptr ref_;           // 参考帧
        Frame::Ptr curr_;          // 当前帧

        cv::Ptr<cv::ORB> orb_;


        VisualOdometry();
        ~VisualOdometry();

    private:

    };
}

#endif //SLAMPREFONT_VISUALODOMETRY_H
