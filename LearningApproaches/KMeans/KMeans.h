//
// Created by smher on 18-4-15.
//

#ifndef KMEANS_KMEANS_H
#define KMEANS_KMEANS_H

#include "headers.h"

/*
bool comparePoint2f(const cv::Point a, const cv::Point b)
{
    return a.x < b.x;
}
*/

class KMeans
{
public:
    KMeans(int k = 10, int m = 50, double e = 1.0);
    ~KMeans();

    void calcKMeans(cv::Mat& imgOut, const cv::Mat& imgIn);
    void setRowCol(int r, int c)
    {
        row_ = r;
        col_ = c;
    }

private:
    float calcDist(const cv::Vec3f& a, const cv::Vec3f& b);
    void initCenters(const cv::Mat& img, int r, int c);
    void checkEps(const cv::Mat& img);
    void updateCenters(const cv::Mat& img);             // 更新聚类中心 kmeans_
    void calcClass(const cv::Mat& img);                 // 计算每一个像素的类别, 用于更新classIdx_, count_

    int k_;                         // 用于表示聚类中心数
    int maxIter_;                   // 最大迭代次数
    double eps_, dist_;             // 最下误差，最大距离

    int row_, col_;
    cv::Mat classIdx_;                         // 用于保存当前像素所所属的类别
    bool flag_;                                // 判断是否收敛, 默认为false
    std::vector<cv::Point> centers_;           // 中心的位置, 用于初始化kmeans_
    std::vector<int> count_;                   // 每一类的元素数, 初始时全为零,
    std::vector<cv::Vec3f> kmeans_;            // 用于 保存第k类的中心向量，每一个元素为3 * 1向量
    //std::set<cv::Point, decltype(comparePoint2f)*> centers_(comparePoint2f);
    //std::set<cv::Point, [](const cv::Point2f a, const cv::Point2f& b)->bool{return a.x < b.x;}> tet_;
};

#endif //KMEANS_KMEANS_H
