//
// Created by smher on 18-3-16.
//

//#include "iostream"
#include "system.h"
#include "myslam/PoseGraph.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    cout << "Hello world." << endl;

    // 进行测试
    // 矩阵块, BlockSolverTraits<int (误差项变化维度)_poseDim, int (误差维度)_landmarkDim>
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;

    // 线性方程求解器：稠密增量方程
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();

    // 矩阵块求解器
    Block* solver_ptr = new Block(linearSolver);

    // 梯度下降法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 图模型
    g2o::SparseOptimizer optimzer;
    optimzer.setAlgorithm(solver);   // 设置求解器
    optimzer.setVerbose(true);       // 显示调试输出

    // 添加节点
    myBackEnd::PoseGraphVertex *v1 = new myBackEnd::PoseGraphVertex();
    myBackEnd::PoseGraphVertex *v2 = new myBackEnd::PoseGraphVertex();

    v1->setId(0);
    v2->setId(1);

    v1->setEstimate(Sophus::SE3());
    v2->setEstimate(Sophus::SE3());

    optimzer.addVertex(v1);
    optimzer.addVertex(v2);

    // 添加边
    myBackEnd::PoseGraphEdge *ed1 = new myBackEnd::PoseGraphEdge();
    ed1->setId(0);
    ed1->setVertex(0, optimzer.vertices()[0]);
    ed1->setVertex(1, optimzer.vertices()[1]);

    optimzer.addEdge(ed1);


    // 初始化
    optimzer.initializeOptimization();

    // 迭代一百次
    optimzer.optimize(100);

    // 返回结果，即读取节点的SE3
    Sophus::SE3 estimateAfterOpti = v1->estimate();
    // OR: Vector 6, 即李代数表示，向量
    Sophus::Vector6d estimateVector = (v1->estimate()).log();
    // Or: Matrix 4 * 4
    Sophus::Matrix4d estimateMatrix = (v1->estimate()).matrix();


    return 0;
}
