//
// Created by smher on 18-3-16.
//

#include "myslam/PoseGraph.h"

namespace myBackEnd
{
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    Matrix6d JR_inv(Sophus::SE3 error)
    {
         Matrix6d JR;
        // 注意.so3返回李群即SO3, 所以后面是log()到李代数(3维向量)，然后hat()到反对称矩阵
        JR.block(0, 0, 3, 3) = Sophus::SO3::hat(error.so3().log());    // startRow, startCol, lengthRow, lengthCol
        JR.block(0, 3, 3, 3) = Sophus::SO3::hat(error.translation());
        JR.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);   // = 3 * 3 zero matrix
        JR.block(3, 3, 3, 3) = Sophus::SO3::hat(error.so3().log());    // startRow, startCol, lengthRow, lengthCol

        JR = Matrix6d::Identity() + JR * 0.5;

        return JR;
    }

    PoseGraph::PoseGraph() //: edgePtr_(nullptr)
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
        _estimate = up * _estimate;         // _estimate: Sophus::SE3
    }

    PoseGraphEdge::PoseGraphEdge()
    {
    }

    // 计算雅克比,用于优化: 误差关于优化变量的导数
    void PoseGraphEdge::linearizeOplus()
    {
        Sophus::SE3 v1 = (static_cast<PoseGraphVertex*>(_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<PoseGraphVertex*>(_vertices[1]))->estimate();

        // 求解误差对SE3的雅克比矩阵
        // _error: Vector6d, se3(李代数)
        _jacobianOplusXi = -JR_inv(Sophus::SE3::exp(_error)) * v1.Adj();             // _jacobianOplusXi: Matrix<double, 6, 6>
        _jacobianOplusXj = JR_inv(Sophus::SE3::exp(_error)) * v2.Adj();             // _jacobianOplusXi: Matrix<double, 6, 6>
    }

    // 计算误差，
    void PoseGraphEdge::computeError()
    {
        Sophus::SE3 nodeLeft = (static_cast<PoseGraphVertex*>(_vertices[0]))->estimate();
        Sophus::SE3 nodeRight = (static_cast<PoseGraphVertex*>(_vertices[1]))->estimate();
        // _error: Matrix<double, 6, 1>
        // 误差： 观测量减去估计量
        _error = (_measurement.inverse() * nodeLeft.inverse() * nodeRight).log();   // _measurement: Sophus::SE3
    }
}

