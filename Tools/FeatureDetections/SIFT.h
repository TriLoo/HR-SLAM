//
// Created by smher on 18-4-21.
//

#ifndef FEATUREEXTRACT_SIFT_H
#define FEATUREEXTRACT_SIFT_H

#include "headers.h"

namespace Feature
{
    class FeatureSIFT
    {
    public:
        FeatureSIFT();
        ~FeatureSIFT();

        void calcSIFT(cv::Mat& imgOut, const cv::Mat& imgIn);
    private:
        int level_;          // levels of different resolution(图像的分辨率不同)
        std::vector<double> alphas_;         // numbers of different scale(高斯滤波的方差大小不同)
    };
}

#endif //FEATUREEXTRACT_SIFT_H
