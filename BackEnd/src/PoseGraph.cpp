//
// Created by smher on 18-3-16.
//

#include "myslam/PoseGraph.h"

namespace myBackEnd
{
    PoseGraph::PoseGraph() : edgePtr_ = nullptr
    {}
    PoseGraph::~PoseGraph()
    {}

    PoseGraph& PoseGraph::createPG()
    {
        static PoseGraph pg;
        return pg;
    }

    // Define PoseGraphVertex
    PoseGraphVertex::PoseGraphVertex() : g2o::BaseVertex<3, Sophus::SE3>()
    {
    }

    void PoseGraphVertex::setToOriginImpl()
    {
        //_estimate << 0, 0, 0;
        // Sophus SE3 初始化
        _estimate = Eigen::Vector3d(0, 0, 0);
    }


}
