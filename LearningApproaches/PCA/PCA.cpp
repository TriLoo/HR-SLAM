//
// Created by smher on 18-4-18.
//

#include "PCA.h"

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

    void PCA::calcPCA(cv::Mat &img, const cv::Mat &imgIn)
    {
    }
}

