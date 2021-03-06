//
// Created by smher on 18-4-21.
//

#include "SIFT.h"

using namespace std;
using namespace cv;

#define ATF at<float>
#define CURV_RATIO  10    // 去除较大曲率的点
#define CONTRAST_TH 0.03  // 去除低对比度的点
#define PI 3.1314

namespace Feature
{
    FeatureSIFT::FeatureSIFT(int o, double a, double ct, double cu_t): octaves_(o), alphas_(a), contrast_threshold_(ct), curvature_threshold_(cu_t)
    {}
    FeatureSIFT::~FeatureSIFT()
    {
    }

    void FeatureSIFT::initSIFT(const cv::Mat& img)
    {
        // TODO: Assure that the input is grayscale image
        Mat imgIn, imgTemp;
        int row = img.rows, col = img.cols;

        // Create different scales
        int k = pow(2, 1.0/scales_);
        Size gaussSize(5, 5);                  // size of gaussian filter: 5 * 5
        for (int i = 0; i < octaves_; ++i)
        {
            for (int j = 0; j < scales_ + 3; ++j)   // 每一个Octave生成Scale+3个不同尺度图像
            {
                alpha = pow(2, i - 1) * pow(k, j) * alphas_;

                // Do Gaussian Filtering
                GaussianBlur(imgIn, imgTemp, gaussSize, alpha);   // gaussSize remains unchanged
                PyrGauss_.push_back(imgTemp);
            }

            // Subsample the image
            //resize(imgIn, imgIn, Size(), 0.5, 0.5, INTER_AREA);
            pyrDown(imgIn, imgIn);
            PyrGauss_.push_back(imgIn);
        }

        // Create DoG using above results
        for (int i = 0; i < octaves_; ++i)
            for (int j = 0; j < scales_ + 2; ++j)        // 每一层Octave生层Scale+2个DoG
            {
                int octShift = i * scales_;
                imgTemp = PyrGauss_[octShift + j + 1] -  PyrGauss_[octShift + j];
                PyrDoGs_.push_back(imgTemp);
            }

        // Create extremum index mat
        int index = 0;
        for (int i = 0; i < octaves_; ++i)
        {
            for (int j = 0; j < scales_; ++j)   // 每一层Octave去掉两个DoG，剩下Scale个极值Flag图
            {
                extremums_.push_back(Mat::zeros(PyrDoGs_[index].size(), CV_32FC1));
            }
            index = octaves_ * (scales_ + 2);
        }
    }

    void FeatureSIFT::extractExtremum()
    {
        // To extract the minimum & maximum locations
        int indexA = 0, indexB = 0;
        Mat temp;
        vector<float> NeighborPixels;
        float currPixel;
        int MaxY = 0, MinY = 0;
        for (int i = 0; i < octaves_; ++i)
        {
            // 待处理的数据是Scale_ + 2个DoG图像，输出时Scale个极值信息图
            for (int j = 1; j < scales_ + 1; ++j)   // 每一层octave，生成Scale_个具有最大值信息的图像
            {
                indexA = i * (scales_ + 2) + j;   // Read
                indexB = i * scales_ + j - 1;     // Write
                temp = PyrDoGs_[indexA];
                for (int m = 1; m < temp.rows - 1; ++m)
                {
                    for (int n = 1; n < temp.cols - 1; ++n)
                    {
                        currPixel = temp.ATF(m, n);
                        // current layer
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i-1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i-1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i-1, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i+1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i+1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA].ATF(i+1, j+1));
                        // above layer
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i-1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i-1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i-1, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i+1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i+1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA - 1].ATF(i+1, j+1));
                        // below layer
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i-1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i-1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i-1, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i, j+1));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i+1, j-1));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i+1, j));
                        NeighborPixels.push_back(PyrDoGs_[indexA + 1].ATF(i+1, j+1));

                        //assert(NeighborPixels.size() >= 26);
                        // OR
                        assert(!NeighborPixels.empty());

                        // Determine the maximum
                        sort(NeighborPixels.begin(), NeighborPixels.end());  // in ascending order
                        // if the vector is empty, generate undefined behavior, Similarly Hereinafter
                        if (currPiexl > *NeighborPixels.back())   // const
                            MaxY  = 1;
                        else
                            MaxY = 0;

                        if (currPixel < NeighborPixels.front())
                            MinY = 1;
                        else
                            MinY = 0;

                        // wirte the result back to extremums_
                        extremums_[indexB].ATF(i, j) = MinY | MaxY;

                        // clear the vector
                        NeighborPixels.clear();
                    }
                }
            }
        }
    }

    // 去掉低对比度特征点 & 不稳定的边缘点
    // 极值处的二阶梯度为零
    // Input is DoG images
    void FeatureSIFT::filterExtremum()
    {
        //
        Mat extreMat;  // storing the locations of  extremums
        Mat_<float> currDoG;
        for (int i = 0; i < octaves_; ++i)
        {
            for (int j = 0; j < scales_; ++j)
            {
                int indexA = i * scales + j;   // 用于索引保存极值的Mat
                int indexB = indexA + 1;       // 用于索引DoG图
                // In a row * 1 data array
                // TODO: convert iextremums_[indexA] to CV_8UC1
                findNonZero(extremums_[indexA], extreMat);  // returns the list of locations of non-zero pixels
                currDoG = PyrDoGs_[indexB];
                for (int k = 0; k < extreMat.rows; ++k)
                {
                    Point loc = extreMat.at<Point>(k);  // see the opencv documents' example
                    // remove low contrast
                    if (currDoG(loc.x, loc.y) < CONTRAST_TH)
                    {
                        extremums_[indexA].at<uchar>(loc) = 0;
                        // OR
                        //extremums_[indexA].at<uchar>(loc.x, loc.y) = 0;
                        continue;
                    }

                    // remove big curvature points
                    float Dxx = currMat(x + 1, y) + currMat(x - 1, y) - 2 * currMat(x, y);
                    float Dyy = currMat(x, y + 1) + currMat(x, y - 1) - 2 * currMat(x, y);
                    float Dxy = currMat(x + 1, y+1) + currMat(x-1, y-1) - currMat(x+1, y-1) - currMat(x-1, y+1);
                    float traceD = Dxx + Dyy;
                    float detD = (Dxx * Dyy) - (Dxy * Dxy);
                    float curvature = traceD * traceD / detD;
                    // 曲率过大
                    if (curvature < 0 || curvature >= CURV_RATIO)
                    {
                        extremums_[indexA].at<uchar>(loc) = 0;
                    }
                }
            }
        }
    }

    // Generate Descriptors
    // 结果保存在kpDescrips_
    void FeatureSIFT::generateDescrip()
    {
        Mat currLoc;
        for (int i = 0; i < octaves_; ++i)
        {
            for (int j = 0; j < scales_; ++j)
            {
                int index = i * (scales_ + 3) + j + 1;
                int indexA = i * scales + j;
                Mat_<float> currMat = PyrGauss_[index];
                Mat_<float> currExtremum = extremums_[indexA];
                findNonZero(currExtremum, currLoc);

                for (int k = 0; k < currLoc.rows; ++k)
                {
                    Point loc = currLoc.at<Point>(k);
                    float dx = currMat(loc.x+1, loc.y) -  currMat(loc.x, loc.y);
                    float dy = currMat(loc.x, loc.y+1) -  currMat(loc.x, loc.y);

                    float Orient = atan(dy, dx);
                    float Mag = sqrt(dx * dx + dy * dy);

                    kpGradOrients_.push_back(Orient);
                    kpGradMags_.push_back(Mag);
                }
            }
        }
        // TODO generate descriptors
    }

    void FeatureSIFT::calcSIFT(cv::Mat &imgOut, const cv::Mat &imgIn)
    {
        initSIFT(imgIn);
        extractExtremum();
        filterExtremum();
        generateDescrip();

        // Draw all keypoints
    }
}

