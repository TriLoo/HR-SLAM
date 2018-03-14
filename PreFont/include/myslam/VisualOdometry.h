//
// Created by smher on 18-3-10.
//

#ifndef SLAMPREFONT_VISUALODOMETRY_H
#define SLAMPREFONT_VISUALODOMETRY_H

#include "system.h"
#include "myslam/Map.h"
#include "myslam/frame.h"

namespace myFrontEnd
{
    class VisualOdometry
    {
    public:         // variables
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
        std::vector<cv::Point3f> pts_3d_ref_;   // 3d points in reference frame
        std::vector<cv::KeyPoint> keypoints_curr_;
        cv::Mat descriptors_curr_;
        cv::Mat descriptors_ref_;
        std::vector<cv::DMatch> feature_matches_;

        Sophus::SE3 T_c_r_estimated_;   // the estimated pose of current frame
        int num_inliers_;
        int num_lost_;
        double map_point_erase_ratio_;

        // parameters
        int num_of_features_;   // number of features
        double scale_factor_;
        int level_pyramid_;
        float match_ratio_;
        float max_num_lost_;
        int min_inliers_;

        double key_frame_min_rot;
        double key_frame_min_tras;


    public:           // Functions
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame);

    protected:   // functions uses the variables in this class
        void extractKeyPoints();
        void computeDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void setRef3DPoints();

        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();

        void optimizeMap();
    };
}

#endif //SLAMPREFONT_VISUALODOMETRY_H
