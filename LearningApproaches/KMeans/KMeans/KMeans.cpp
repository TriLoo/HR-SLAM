//
// Created by smher on 18-4-15.
//

#include "KMeans.h"

#define ATV at<Vec3f>
#define ATF at<float>

#define PRINTDIST cout << currDist << endl

using namespace std;
using namespace cv;

KMeans::KMeans(int k, int m, double e):k_(k), maxIter_(m), eps_(e), flag_(false)
{
}

KMeans::~KMeans()
{
}

float KMeans::calcDist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    Vec3f sub = a - b;

    Vec<float, 1> tempSum = sub.t() * sub;
    double sum = static_cast<double>(tempSum[0]);

    return tempSum[0];
}

// TODO: use this check during calculation
void KMeans::checkEps(const cv::Mat &img)
{
    flag_ = true;

    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            int k = floor(classIdx_.at<float>(i, j));
            if (calcDist(img.ATV(), kmeans_[k]) > eps_)
                flag_ = false;
        }
}

void KMeans::initCenters(const cv::Mat &img, int r, int c)
{
    assert(r >= 1 && c >= 1);
    classIdx_ = Mat::zeros(img.size(), CV_32FC1);

    RNG rd;

    centers_.clear();
    while(centers_.size() < k_)
    {
        int rowIdx = rd.uniform(0, row_);      // return [0, row_)
        int colIdx = rd.uniform(0, col_);

        Point2f temp(colIdx, rowIdx);

        // 防止重复元素的存在
        if (find_if(centers_.begin(), centers_.end(), [temp](const Point& a){return a.x == temp.x && a.y == temp.y;}) != centers_.end())
            continue;

        centers_.push_back(Point2f(colIdx, rowIdx));
        kmeans_.push_back(img.ATV(rowIdx, colIdx));
        count_.push_back(0);
    }
    // Output the k_ initial centers
    //for (const auto& ele : centers_)
        //cout << ele << endl;
}

void KMeans::updateCenters(const cv::Mat& img)
{
    const int DIM = kmeans_.size();
    vector<Vec3f> tempSum(DIM, Vec3f(0, 0, 0));
    for (int i = 0; i < row_; ++i)
    {
        for (int j = 0; j < col_; ++j)
        {
            int idx = floor(classIdx_.at<float>(i, j));
            assert(idx >= 0 && idx < DIM);
            tempSum.at(idx) += img.ATV(i, j);
        }
    }

    for (int j = 0; j < DIM; ++j)
    {
        int a = count_.at(j);
        kmeans_.at(j) = tempSum.at(j) / (a == 0 ? 1 : a);       //需要判断除数是否为零! ! !
    }
}

void KMeans::calcClass(const cv::Mat &img)
{
    int idx = 0;         // 当前最近的类中心的索引，一个整数
    float dist = 0.0;    // 当前最小的距离
    float currDist = 0.0;

    assert(kmeans_.size() == k_);

    // 将每一类的数量先值为零, see: https://stackoverflow.com/questions/8848575/fastest-way-to-reset-every-value-of-stdvectorint-to-0
    memset(&count_[0], 0, count_.size() * sizeof(count_[0]));          // Fastest
    //count_.assign(count_.size(), 0);
    //fill(count_.begin(), count_.end(), 0);

    Vec3f a, b;
    for (int i = 0; i < row_; ++i)
    {
        for (int j = 0; j < col_; ++j)
        {
            a = img.ATV(i, j);            // Caution: here have been writen as ATF wrongly
            b = kmeans_.at(0);
            idx = 0;
            currDist = calcDist(a, b);
            // PRINTDIST;

            for (int k = 1; k < k_; ++k)
            {
                b = kmeans_.at(k);
                dist = calcDist(a, b);

                if (dist < currDist)
                {
                    currDist = dist;
                    idx = k;              // 当前值
                }
            }

            classIdx_.ATF(i, j) = idx;
            count_.at(idx)++;
        }
    }
}

void KMeans::calcKMeans(cv::Mat& imgOut, const cv::Mat &imgIn)
{
    RNG rd;
    int chann = imgIn.channels();
    setRowCol(imgIn.rows, imgIn.cols);

    // init centers
    initCenters(imgIn, imgIn.rows, imgIn.cols);
    //cout << "Step 1. Success." << endl;

    for (int i = 0; i < maxIter_ && (!flag_); ++i)
    {
        // 计算所属的类中心
        calcClass(imgIn);           // the results are stored in 'count_', 'classIdx_'
        //cout << "Step 2. Success." << endl;
        // 更新类中心,
        updateCenters(imgIn);       // based on 'count_', 'classIdx_'
        //cout << "Step 3. Success." << endl;
    }

    // Prepare the output
    for (const auto& ele : count_)
        cout << ele << endl;

    imgOut = Mat::zeros(imgIn.size(), CV_32FC3);

    Mat lut = Mat_<Vec3f>(5, 1);
    lut.at<Vec3f>(0, 0) = cv::Vec3f(1, 0.2, 0.2);
    lut.at<Vec3f>(1, 0) = cv::Vec3f(0.2, 1.0, 0.0);
    lut.at<Vec3f>(2, 0) = cv::Vec3f(0.4, 0.5, 1);
    lut.at<Vec3f>(3, 0) = cv::Vec3f(0.6, 0.3, 0.2);
    lut.at<Vec3f>(4, 0) = cv::Vec3f(0.8, 0.1, 1);

    for (int i = 0; i < row_; ++i)
    {
        for (int j = 0; j < col_; ++j)
        {
            int t = classIdx_.at<float>(i, j);
            imgOut.at<Vec3f>(i, j) = lut.at<Vec3f>(t, 0);
        }
    }
    //cout << "Step 4. Success." << endl;
}

#undef ATV
