//
// Created by smher on 18-5-20.
//

#ifndef FEATUREEXTRACT_HOG_H
#define FEATUREEXTRACT_HOG_H

#include "headers.h"

namespace Feature
{
    /*
     * 计算HoG特征时，与SIFT最主要的特点是:
     *   a. 不需要进行多尺度分解
     *   b. 只需要进行局部一个Cell内部进行Normalize, 而不是一整幅图, 这让它查找复杂背景下的曲线轮廓更有优势
     *   c. 这里有时间再写吧
     */
    class FeatureHoG
    {
    public:
        HoG();
        ~HoG();

        void calcHoG(cv::Mat& imgOut, const cv::Mat& imgIn);

    private:

    };
}

#endif //FEATUREEXTRACT_HOG_H
