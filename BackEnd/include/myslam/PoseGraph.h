//
// Created by smher on 18-3-16.
//

#ifndef BACKEND_POSEGRAPH_H
#define BACKEND_POSEGRAPH_H

#include "system.h"

class PoseGraphEdge;

namespace myBackEnd
{
    class PoseGraph
    {
    public:
        typedef std::shared_ptr<PoseGraph> Ptr;

        ~PoseGraph();

        // 单例模式
        PoseGraph& createPG();

    private:
        PoseGraph();   // 单例模式
        std::shared_ptr<PoseGraphEdge> edgePtr_;        // Has a Pose Graphic Edge

    };

    // 定义节点类型，基于李代数
    // 定义位姿图，基于G2O, 位姿的李代数表示的维度为：3， 类型为李代数: SE3
    // 主要定义： setToOriginImpl()以及oplusImpl()
    class PoseGraphVertex : public g2o::BaseVertex<3, Sophus::SE3>
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PoseGraphVertex();

        bool read(std::istream& is);
        bool write(std::ostream& os) const;

        virtual void setToOriginImpl();
        virtual void oplusImpl(const double * update_);
    };

    // 定义边类型
    // 双节点边. error vector dimension: 3, measurement type: Sophus::SE3 or Eigen::Vector3d, VertexXi Type: PoseGraphVertex, VertexXj Type: PoseGraphVertex
    // 主要定义误差函数与Jacobin函数
    //class PoseGraphEdge : public g2o::BaseBinaryEdge<3, Sophus::SE3, PoseGraphVertex, PoseGraphVertex>
    class PoseGraphEdge : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, PoseGraphVertex, PoseGraphVertex>
    {
        PoseGraphEdge();

        bool read(std::istream& is){}
        bool write(std::ostream& os) const {}

        virtual void linearizeOplus();
        void computeError();
    };

}

#endif //BACKEND_POSEGRAPH_H
