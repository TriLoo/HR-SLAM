//
// Created by smher on 18-4-21.
//

#include "SIFT.h"

using namespace std;
using namespace cv;

namespace Feature
{
    FeatureSIFT::FeatureSIFT(int o, double a, double ct, double cu_t): octaves_(o), alphas_(a), contrast_threshold_(ct), curvature_threshold_(cu_t)
    {}
    FeatureSIFT::~FeatureSIFT()
    {
    }

    void FeatureSIFT::initSIFT(const cv::Mat& img)
    {
        Mat imgIn, imgTemp;

        int k = pow(2, 1.0/scales_);
        Size gaussSize(5, 5);                  // size of gaussian filter: 5 * 5
        for (int i = 0; i < octaves_; ++i)
        {
            for (int j = 0; j < scales_; ++j)
            {
                alpha = pow(2, i - 1) * pow(k, j) * alphas_;

                // Do Gaussian Filtering
                GaussianBlur(imgIn, imgTemp, gaussSize, alpha);   // gaussSize remains unchanged
                PyrGauss_.push_back(imgTemp);
            }

            // Subsample the image
            resize(imgIn, imgIn, Size(), 0.5, 0.5, INTER_AREA);
        }
    }

    void FeatureSIFT::calcSIFT(cv::Mat &imgOut, const cv::Mat &imgIn)
    {
    }
}

