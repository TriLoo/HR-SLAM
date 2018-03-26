//
// Created by smher on 18-3-16.
//

#include "myslam/PoseGraph.h"

namespace myBackEnd
{
    PoseGraph::PoseGraph() : edgePtr_(nullptr)
    {}
    PoseGraph::~PoseGraph()
    {}

    PoseGraph& PoseGraph::createPG()
    {
        static PoseGraph pg;
        return pg;
    }

    // 定义节点
    // Define PoseGraphVertex Constructor
    PoseGraphVertex::PoseGraphVertex() : g2o::BaseVertex<6, Sophus::SE3>()
    {
    }

    void PoseGraphVertex::setToOriginImpl()
    {
        //_estimate << 0, 0, 0;
        // Sophus SE3 初始化
        //_estimate = Eigen::Vector3d(0, 0, 0);
        _estimate = Sophus::SE3();
    }

    void PoseGraphVertex::oplusImpl(const double *update_)
    {
        Sophus::SE3 up(Sophus::SO3(update_[3], update_[4], update_[5]), Sophus::Vector3d(update_[0], update_[1], update_[2]));
        _estimate = up * _estimate;
    }
}

