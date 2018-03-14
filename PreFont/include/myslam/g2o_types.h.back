//
// Created by smher on 18-3-13.
//

#ifndef SLAMPREFONT_G2O_TYPES_H
#define SLAMPREFONT_G2O_TYPES_H

#include "system.h"
#include "myslam/camera.h"

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/core/robust_kernel.h"
#include "g2o/core/robust_kernel_impl.h"


namespace myFrontEnd
{
    class EdgeProjectXYZ2UVPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        virtual void computeError();
        virtual void linearizeOplus();

        virtual bool read(std::istream& is) {}
        virtual bool write(std::ostream& os) const {}

        Eigen::Vector3d point_;
        Camera* camera_;
    private:
    };
}

#endif //SLAMPREFONT_G2O_TYPES_H
