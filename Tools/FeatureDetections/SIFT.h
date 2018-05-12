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
        FeatureSIFT(int o = 3, double a = 0.5, double ct = 0.03, double cu_t = 10.0);
        ~FeatureSIFT();

        void calcSIFT(cv::Mat& imgOut, const cv::Mat& imgIn);
    private:
        int octaves_;          // levels of different resolution(图像的分辨率不同), 即Octave == (S)
        int scales_;           // number of different scales (alpha in Gaussian Filter)
        double alphas_;         // numbers of different scale(高斯滤波的方差大小不同),此外，k = 2^{1/s}
        double contrast_threshold_, curvature_threshold_;      // 阈值：对比度阈值，默认为0.03；曲率阈值，默认为10.

        // 保存中间结果，如 Image Octaves
        std::vector<cv::Mat> PyrGauss_;
        std::vectro<cv::Mat> PyrDoGs_;

        // 保存特征点信息：位置、尺度、梯度方向及大小等
        std::vector<cv::Point2f> keypoints_;                   // 二维特征点坐标, use erase() to remove bad points
        std::vector<int> kpScales_;                            // 特征点所处的尺度
        std::vector<double> kpGradOrients_;                    // 特征点的梯度方向
        std::vector<double> kpGradMags_;                       // 特征点梯度幅值

        // 保存描述子: 每一个特征点对应一个128维的向量
        cv::Mat kpDescrips_;

        // Some Operations to Complete 'calcSIFT(...)'
        void initSIFT(const cv::Mat& img);
    };
}

#endif //FEATUREEXTRACT_SIFT_H
