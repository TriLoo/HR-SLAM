//
// Created by smher on 18-3-13.
//

// Caution: the viz lib is not installed in my PC for now, so the functionality of this program have not been verified.

#include "system.h"
#include "fstream"

#include "myslam/Config.h"
#include "myslam/VisualOdometry.h"

#include "opencv2/viz.hpp"

using namespace std;
using namespace cv;
//using namespace myFrontEnd;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: VO configre.file" << endl;
        return 1;
    }

    myFrontEnd::Config::setParameterFile(argv[1]);

    myFrontEnd::VisualOdometry::Ptr vo(new myFrontEnd::VisualOdometry());

    string dataset_dir = myFrontEnd::Config::getParam<string>("dataset_dir");
    cout << "Dataset: " << dataset_dir << endl;
    ifstream fin(dataset_dir + "/associate.txt");

    if (!fin)
    {
        cerr << "Open dataset directory failed." << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while(fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file ;
        rgb_times.push_back(stod(rgb_time));
        rgb_files.push_back(rgb_file);
        depth_times.push_back(stod(depth_time));
        depth_files.push_back(depth_file);

        if (fin.good() == false)
            break;
    }

    myFrontEnd::Camera::Ptr camera(new myFrontEnd::Camera);

    // Visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);   // 世界坐标系，相机坐标系
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    cout << "Read total " << rgb_files.size() << " entries." << endl;

    for (int i = 0; i < rgb_files.size(); ++i)
    {
        Mat color = imread(rgb_files[i]);
        Mat depth = imread(depth_files[i], IMREAD_UNCHANGED);

        if (color.empty() || depth.empty())
            continue;

        myFrontEnd::Frame::Ptr pFrame = myFrontEnd::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->timeStamp_ = rgb_times[i];

        vo->addFrame(pFrame);

        if (vo->state_ == myFrontEnd::VisualOdometry::LOST)
            break;

        Sophus::SE3 Tcw = pFrame->Tcw_.inverse();     // From Camera to World

        // show the map the camera pose
        cv::Affine3d M(
                cv::Affine3d::Mat3(
                Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1), Tcw.rotation_matrix()(0,2),
                Tcw.rotation_matrix()(1,0), Tcw.rotation_matrix()(1,1), Tcw.rotation_matrix()(1,2),
                Tcw.rotation_matrix()(2,0), Tcw.rotation_matrix()(2,1), Tcw.rotation_matrix()(2,2)),
                cv::Affine3d::Vec3(Tcw.translation()(0, 0), Tcw.translation()(1, 0), Tcw.translation()(2, 0))
        );

        imshow("Image", color);
        cv::waitKey(1);
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}
