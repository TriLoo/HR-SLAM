# Search By Projection

smh

2018.11.24

## 简介

`SearchByProjection(args)`是一个重载函数，共重载了4次。

## 相邻帧之间的search by projection

### 函数原型

```c++
int ORBMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);
```

### 原理

1. 从相机矩阵恢复相机中心在世界坐标系的坐标

   这里假设第一帧时相机位于世界坐标系原点。第二帧对应的相机矩阵： $[\mathbf{R} | \mathbf{t}]$

   ​    // 第一：相机坐标等价于相加位于世界坐标系原点、且朝向世界坐标系的Z轴(可通过针孔相机模型的内参推导过程看出)

   ​    // 第二：所以相机的中心在世界坐标系的位置就是K[R|t]中的t

   ​    // 第三：而世界坐标系到相机坐标系的过程是Xc = R(Xw - Cc)

   ​    // 第四：所以t = -RCc，其中Cc即为要求解的相机在世界坐标系的坐标

   ​    // 第五：旋转矩阵R是正交的，即转置等价于取逆

   所以可以得到一下代码了：

   ```c++
   const cv::Mat twc = -Rcw.t() * tcw;
   ```

2.  `SearchByProjection`目的是搜索上一帧的关键点在当前帧的匹配关键点，用到的方法就是重投影的方法

   1.  首先将上一帧的关键点得到对应MapPoint的世界坐标

   2.  然后将该世界坐标投影到当前帧上，计算其投影点的**像素坐标**

   3.  下一步就是在这个重投影点的一定的范围内进行搜索，搜索范围与特征点计算的octive有关，且搜索形状为$2r * 2r$的方形

   4.  最后一步是对搜索得到的所有匹配点进行旋转角度一致性的测试，也就是假设大部分的关键点的旋转角度应该等于图像的旋转角度，图像的旋转角度就是所有所有关键点的旋转角度的三个最大的旋转方向

   5.  在确定范围的时候，使用了一个trick，就是使用基于Grid的方式来进行搜索，Grid的在`Frame`里面的数据结构是:

      ```c++
      vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
      ```

      这个在Grid中的cell搜索的函数是`GetFeatureInArea`，同时在搜索的过程中，判断距离使用的是两个分量(x, y)方向的绝对值距离。



### 计算过程

计算过程详见代码中的注释。

