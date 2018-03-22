//
// Created by smher on 18-3-20.
//

#ifndef BACKEND_SYSTEM_H
#define BACKEND_SYSTEM_H

#include "iostream"
#include "vector"
#include "string"
#include "memory"
#include "stdexcept"
#include "algorithm"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"


#include "Eigen/Core"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "sophus/se3.h"
#include "sophus/so3.h"


#endif //BACKEND_SYSTEM_H
