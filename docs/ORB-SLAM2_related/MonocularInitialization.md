# MonocularInitialization计算过程

smh

2018.10.23

## 坐标正则化

坐标正则化的过程比较简单，主要就是下面的三个步骤：

1.  计算所有特征点的中心坐标.
2.  计算所有特征点与中心点的平均距离，距离就是差值的绝对值.
3.  所有的特征点的坐标减去中心点的坐标，然后除以到中心点的平均距离.

代码略。



## 计算H矩阵

函数的主要步骤：

1.  坐标Normalize
2.  定义每次迭代所需要的变量，包括特征点的位置索引、当次迭代的内点数等
3.  循环`mMaxIterations`次
4.  调用`ComputeH21`使用**DLT**算法计算H矩阵
5.  调用`CheckHomography`计算当次迭代计算得到H的得分
6.  如果当次迭代得到的H的得分(内点个数)较多，那么就更新参数，保存当前的H、分数、属于内点的匹配点等。

### ComputeH21

其实，这里主要就是两个主要步骤：(1)计算矩阵A，每一对匹配点可以提供A的两行，每行9个元素;(2) 就是对矩阵A进行SVD分解了，然后要求解的H就是最小的奇异值对应的V矩阵的一列，也就是VT的一行。具体的细节可以很容易从下面的代码中看出来，用到的主要理论就是MVG书中第89页的式(4.3)。

```c++
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();   // 等于8

    cv::Mat A(2*N,9,CV_32F);    // 每个点提供两个等式、H矩阵共有9个变量需要确定

    // 全部8个点后，A矩阵就是MVG书中第89页中的式(4.3)
    for(int i=0; i<N; i++)
    {
        // 为什么这里要把临时变量定义成const的 !？
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    // MODIFY_A: 表示，可以修改输入矩阵，以节省空间、加速处理过程等
    // FULL_UV : U and VT will be full-size square orthogonal matrices.
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 计算的H矩阵就是VT的最后一行，然后reshape到3*3的矩阵大小
    // 0 作为reshape的channel参数，表示channel维度会保持不变
    return vt.row(8).reshape(0, 3);
}
```



### CheckHomography

这个函数主要就是计算使用H矩阵作为变换矩阵时的一个得分情况。具体的得分计算包括了正向、反向投影误差在一定范围内的特征点。具体来说，主要包括五个步骤：

1.  计算第二幅图像的特征点反向投影到第一幅图像中
2.  计算投影过来的点与该点在第一幅图像中的匹配点之间的距离
3.  计算距离的平方欧式距离，并除以sigma的平方，计算结果保存在$chiSquare$
4.  第三步的计算结果与一个阈值之间比较，若大于阈值，则该点属于外点；若小于阈值，则score就加上$th - chiSquare$，其中$th$为预先定义好的一个阈值
5.  对于第一幅图像特征点的正向投影重复上述计算过程

具体的代码实例如下：

```c++
// Reprojection error in first image
        // x2in1 = H12*x2

        // 下面是计算第二幅图像重投影到第一幅图像上的坐标计算
        // 先计算重投影后的低三个维度的坐标W2in1inv
        // 然后计算重投影到第一幅图像的坐标的前两位
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);   
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影后的点与原始的点之间的欧式距离(平方)
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 上面的结果除以sigma * sigma
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;   // score是阈值与chiSquare之间的距离

        // Reprojection error in second image
        // x1in2 = H21*x1
        ...   // 后面省略
```



## 计算F矩阵

计算F矩阵的过程与计算H矩阵的过程大体一致，但具体的计算与H的计算有些差别，主要的差别主要由**TODOs**引起。



## 使用H进行重建



## 使用F进行重建

