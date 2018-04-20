//
// Created by smher on 18-4-18.
//

#include "PCA.h"

using namespace std;
using namespace cv;

namespace LA
{
    myPCA::myPCA(int k):K_(k) {}
    myPCA::~myPCA() {}

    void myPCA::setNumK(int k)
    {
        assert(k >= 1);
        K_ = k;
    }

    int myPCA::getNumK()
    {
        return K_;
    }

    // 根据DL第二章的例子，主成份对应输入数据的协方差矩阵的最大奇异向量
    void myPCA::calcPCA(cv::Mat &imgOut, const cv::Mat &imgIn)
    {
        Mat inImg = imgIn;
        if (inImg.channels() == 3)
            cvtColor(inImg, inImg, CV_BGR2GRAY);
        if (inImg.type() == CV_8UC1)
            inImg.convertTo(inImg, CV_32FC1, 1.0);

        // Step 1: 中心化,计算每一列的均值，然后减去该均值
        // One way: use .at<float> to fetch all elements
        // Second way: use Mat::col etc.
        // Alternatives: use reduce, repeat

        // 2nd
        const int row = imgIn.rows;
        const int col = imgIn.cols;
        Mat currCol;
        /*
        for (int i = 0; i < col; ++i)
        {
            currCol = inImg.col(i);             //
            // for test
            if(i == 1)
                cout << "Size returned by col():(row * col) " << currCol.rows << " * " << currCol.cols << endl;


            Scalar tempMean = mean(currCol);

            inImg.col(i) = inImg.col(i) - tempMean;           // 中心化
        }
         */

        // 3th, 等价于2nd 方法
        Mat colMean;
        reduce(inImg, colMean, 0, REDUCE_SUM);         // 0 means that the matrix is reduced to a single row
        colMean /= inImg.rows;
        /*
        cout << "Before repeat: " << colMean.rows << " * " << colMean.cols << endl;
        Mat tempMatIn = repeat(inImg, col, 1);
        cout << "After repeat: " << tempMatIn.rows << " * " << tempMatIn.cols << endl;
        inImg -= tempMatIn;
        */
        inImg -= repeat(colMean, col, 1);                // nx, ny >= 1

        // 因为默认输入中每一行代表一个样本，所以这里转置放在后面了
        Mat covMat = inImg * inImg.t();

        // Step 2: calculate eigenvectors
        Mat eigenValues, eigenVectors;
        eigen(covMat, eigenValues, eigenVectors);             // 注意：这里的eigenVectors是以行的形式存储特征向量的
        normalize(eigenVectors, eigenVectors, 0.0, 1.0, NORM_MINMAX);

        // Step 3: calculate the dimension-reduced result
        // 注意：这里的输入数据是转置后的，即每一行代表一个样本，因此降维后的结果: X * D, 其中D的每一列为一个特征向量, 输出的样本也是每一行为一个样本
        //      等价于DL书中：输入数据未转置，即每一列代表一个样本，则降维后结果为： D^T * X, 输出的结果也是每一列为一个样本

        assert(K_ >= 1);
        imgOut = eigenVectors.rowRange(0, K_) * inImg.t();          // 在第二种情形下的修改, 此时，输出是一个K_ * imgIn.rows的矩阵 ! ! !
    }
}

