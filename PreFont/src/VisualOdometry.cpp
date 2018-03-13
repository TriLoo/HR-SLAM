//
// Created by smher on 18-3-10.
//

#include "myslam/VisualOdometry.h"
#include "myslam/Config.h"
#include "myslam/g2o_types.h"

using namespace std;
using namespace cv;

namespace myFrontEnd
{
    VisualOdometry::VisualOdometry() : state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map),
                                       num_lost_(0), num_inliers_(0)
    {
        num_of_features_     = Config::getParam<int>("number_of_fetures");
        scale_factor_        = Config::getParam<double>("scale_factor");
        level_pyramid_       = Config::getParam<int>("level_pyramid");
        match_ratio_         = Config::getParam<float>("match_ratio");
        max_num_lost_        = Config::getParam<float>("max_num_lost");
        min_inliers_         = Config::getParam<int>("min_inliers");
        key_frame_min_rot    = Config::getParam<double>("keyframe_rotation");
        key_frame_min_tras   = Config::getParam<double>("keyframe_translation");

        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);       //
    }
    VisualOdometry::~VisualOdometry()
    {
    }

    void VisualOdometry::extractKeyPoints()
    {
        orb_->detect(curr_->color_, keypoints_curr_, cv::Mat());
    }

    void VisualOdometry::computeDescriptors()
    {
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
    }

    void VisualOdometry::featureMatching()
    {
        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(descriptors_curr_, descriptors_ref_, matches, cv::Mat());

        float min_dis = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2)
        {return m1.distance < m2.distance;})->distance;

        feature_matches_.clear();

        for (auto & m : matches)
        {
            if (m.distance < max<float>(min_dis * match_ratio_, 30.0))
            {
                feature_matches_.push_back(m);
            }
        }
        cout << "Good Matches: " << feature_matches_.size() << endl;

        /*
        // 去除不好的匹配
        // sorts the elements in the range [frist, last) in ascending order
        std::sort(matches.begin(), matches.end());
        int num = matches.size() * match_ratio_;
        matches.erase(matches.begin()+num, matches.end());
        */
    }

    void VisualOdometry::setRef3DPoints()
    {
        pts_3d_ref_.clear();
        descriptors_ref_ = Mat();
        for (size_t i = 0; i < keypoints_curr_.size(); ++i)
        {
            double d = ref_->findDepth(keypoints_curr_[i]);
            if (d > 0)
            {
                Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(
                        Eigen::Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
                );

                pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
                // push_back: add elements to the bottom of the matrix
                descriptors_ref_.push_back(descriptors_curr_.row(i));
            }
        }

        /*
        for (const auto & m : feature_matches_)
        {
            pts_3d_ref_.push_back(curr_->camera_->pixel2camera(keypoints_curr_[m.trainIdx].pt));
        }
        */
    }

    void VisualOdometry::poseEstimationPnP()
    {
        vector<Point2f> pts2d;
        vector<Point3f> pts3d;
        for (const auto& fm : feature_matches_)
        {
            pts3d.push_back(pts_3d_ref_[fm.queryIdx]);
            pts2d.push_back(keypoints_curr_[fm.trainIdx].pt);
        }

        // 相机内参
        Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_, 0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);

        Mat rvec, tvec, inliers;
        solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        cout << "PnP inliers: " << num_inliers_ << endl;
        T_c_r_estimated_ = Sophus::SE3(Sophus::SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
                                        Sophus::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0) ));

        // using BA to optimize the pose
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(T_c_r_estimated_.rotation_matrix(),
                                          T_c_r_estimated_.translation()));
        optimizer.addVertex(pose);

        // edges
        for (int i = 0; i < inliers.rows; ++i)
        {
            int index = inliers.at<int>(i, 0);
            // 3d -> 2d projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Eigen::Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            optimizer.addEdge(edge);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_r_estimated_ = Sophus::SE3(pose->estimate().rotation(),
                                        pose->estimate().translation());
    }

    bool VisualOdometry::checkEstimatedPose()
    {
        if (num_inliers_ < min_inliers_)
        {
            cout << "Reject because inlier is too small: " << num_inliers_ << endl;
            return false;
        }

        Sophus::Vector6d d = T_c_r_estimated_.log();

        // 运动过大
        if (d.norm() > 5.0)
        {
            cout << "Reject because motion is too large: " << d.norm() << endl;
            return false;
        }

        return true;
    }

    bool VisualOdometry::checkKeyFrame()
    {
        Sophus::Vector6d d = T_c_r_estimated_.log();
        Sophus::Vector3d trans = d.head<3>();
        Sophus::Vector3d rot = d.tail<3>();
        if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_tras)
            return true;
        return false;
    }

    void VisualOdometry::addKeyFrame()
    {
        cout << "Adding a key-frame" << endl;
        map_->insertKeyFrame(curr_);
    }

    bool VisualOdometry::addFrame(Frame::Ptr frame)
    {
        switch (state_)
        {
            case INITIALIZING:
            {
                state_ = OK;
                curr_ = ref_ = frame;
                map_->insertKeyFrame(frame);
                extractKeyPoints();
                computeDescriptors();
                setRef3DPoints();
                break;
            }
            case OK:
            {
                curr_ = frame;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                poseEstimationPnP();
                if (checkEstimatedPose() == true)
                {
                    curr_->Tcw_= T_c_r_estimated_ * ref_->Tcw_;
                    ref_ = curr_;
                    setRef3DPoints();
                    num_lost_ = 0;
                    if (checkKeyFrame() == true)
                    {
                        addKeyFrame();
                    }
                } else
                {
                    num_lost_++;
                    if (num_lost_ > max_num_lost_)
                        state_ = LOST;
                    return false;
                }
                break;
            }
            case LOST:
            {
                cout << "VO has lost" << endl;
                break;
            }
        }
        return true;
    }
}

