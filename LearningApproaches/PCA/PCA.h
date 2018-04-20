//
// Created by smher on 18-4-18.
//

#ifndef PCA_PCA_H
#define PCA_PCA_H

#include "headers.h"

namespace LA         // Learning Approaches namespace
{
    class myPCA
    {
    public:
        myPCA(int k = 1);
        myPCA(const myPCA& a) = delete;
        myPCA& operator=(const myPCA& a) = delete;
        ~myPCA();

        void setNumK(int k);
        int getNumK();

        void calcPCA(cv::Mat& img, const cv::Mat& imgIn);
    private:
        int K_;           // the first k-th components
    };
}

#endif //PCA_PCA_H
