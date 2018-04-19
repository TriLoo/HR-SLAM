//
// Created by smher on 18-4-18.
//

#include "PCA.h"

using namespace std;
using namespace cv;

namespace LA
{
    PCA::PCA(int k):K_(k) {}
    PCA::~PCA() {}

    void PCA::setNumK(int k)
    {
        K_ = k;
    }

    int PCA::getNumK()
    {
        return K_;
    }

    // 根据DL第二章的例子，主成份对应输入数据的协方差矩阵的最大奇异向量
    void PCA::calcPCA(cv::Mat &img, const cv::Mat &imgIn)
    {
        Mat inImg = imgIn;
        if (inImg.channels() == 3)
            cvtColor(inImg, inImg, CV_BGR2GRAY);
        if (inImg.type() == CV_8UC1)
            inImg.convertTo(inImg, CV_32FC1, 1.0);

        // Step 1: 中心化,计算每一列的均值，然后减去该均值
        // Alternatives: use reduce, repeat

    }
}

