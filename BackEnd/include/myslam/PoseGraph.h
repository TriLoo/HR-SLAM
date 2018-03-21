//
// Created by smher on 18-3-16.
//

#ifndef BACKEND_POSEGRAPH_H
#define BACKEND_POSEGRAPH_H

#include "system.h"

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
    };

    // 定义节点类型，基于李代数
    // 定义位姿图，基于G2O, 位姿的李代数表示的维度为：3， 类型为李代数: SE3
    class PoseGraphVertex : public g2o::BaseVertex<3, Sophus::SE3>
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PoseGraphVertex();

        bool read(std::istream& is);
        bool write(std::ostream& os) const;

        // TODO

    };

}

#endif //BACKEND_POSEGRAPH_H
