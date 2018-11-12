# ORB-SLAM2代码详解笔记

参考：[泡泡机器人-ORBSLAM源码详解-吴博](https://v.qq.com/iframe/preview.html?width=500&height=375&auto=0&vid=y0344ko90m0)

smh

2018.11.12

## 代码主要结构

* 三个线程

  1.  Tracking

  2.  Local Mapping

     与Tracking之间的关联是通过KeyFrame来实现的。

  3.  Loop Closing

     输入数据来自Local Mapping中删除冗余关键帧之后的关键帧数据。

* 命名规则：

  "p" 表示指针类型；"n"表示int类型；"b" 表示bool类型；"s"表示set类型；"v"表示vector类型；"l"表示list类型；"m"表示类成员变量。

## System 入口

​	输入图像$\rightarrow$ 转为灰度图， 构造Frame，进入Track线程。

​	mpIniORBextractor相比mpORBextractorLeft提取的特征点多一倍。

## Frame.cpp

* 双目立体匹配

  * 对于左目每个特征点建立**带状**搜索区域
  * 通过描述子进行特征点匹配
  * SAD划窗法得到匹配修正量

*  Disparity与Depth(RGB-D)

  *  $$d = x - x' = \frac{bf}{Z}$$

    其中，d表示disparity，b为baseline的长度，单位为m，其它变量(b, d)是像素单位，所以Z是m为单位。

## Tracking线程

* 初始化

  立体、单目。

* 相机位姿跟踪

  mbOnlyTracking(true)： 同时跟踪与定位，不插入关键帧，局部地图不工作，所以后面的Loop Clouser也不工作。

  * 位姿跟踪

    分两种： TrackWithMotionModel()， TrackReferenceKeyFrame()

  * Relocalization()， 重定位

* 局部地图跟踪

  * UpdateLocalMap()

    更新Local Key Frame，更新局部点(Local Points)

  * SearchLocalPoints()

    获得局部地图与当前帧的匹配。

  * PoseOptimization()

    最小化投影误差优化位姿。

* 是否生成关键帧

  * 很长时间没有插入关键帧、局部地图空间、跟踪即将失败
  * 跟踪地图MapPoints的比例比较少

* 生成关键帧

  * KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB)

    mCurrentFrame表示当前帧，其中mpMap为总的地图管理器。mpKeyFrameDB用于保存之前的数据。

  * 对于双目或RGBD摄像头构造一些MapPoints，为MapPoints添加属性

    这些MapPoints只是为了跟踪，不会添加到地图中。



### 初始化

单目Initializer.cpp

通过八点法+DLT求解H矩阵、F矩阵。

* 首先对坐标进行归一化：减去均值、除以一阶矩的均值

### 相机位姿跟踪

#### TrackWithMotionModel

​	适用于匀速模型。$\Delta R_k \simeq \Delta R_{k-1}$。根据这个假设可以预测第k时刻的位姿，然后在利用重投影误差进行优化，得到最终的位姿。其中$R$表示相机位姿。

​	**这里是不是可以引入IMU来测量相对旋转呢？？？**

#### TrackReferenceKeyFrame

​	当TrackWithMotionModel失败后，使用此方法，具体实现是，使用距离自己最近的参考帧的位姿当做当前帧的位姿即：

$$SE_{k} = SE_{KF} $$

​	同样是把这个位姿当做初始估计值，然后后续还需要进行重投影误差优化。

#### Relocation

​	使用EPnP求解。**TODO**.

### 插入MapPoints的地方

* 初始化时
* 插入关键帧时
* LocalMapping时

## Local Mapping线程

该线程的数据来源是Tracking线程中的KeyFrame，送到一个链表中。

* 检查队列

  * CheckNewKeyFrames()

    即检查上面的链表里面是否还有关键帧，如果没有关键帧就阻塞，如果有的话就继续处理。

*  处理新关键帧

  ProcessNewKeyFrame()

  * 更新MapPoints与KeyFrame的关联

    相互关联。

  * UpdateConnections()

    跟踪相连帧，更新共视关系，即观测到MapPoints。维护一个扩展树。

* 提出MapPoints

  * 提出地图中新添加的但质量不好的MapPoints或冗余

    IncreaseFound / Increase Visible < 25%，表示应该被观测到，但实际观测到比较少，就会提出这个MapPoints

    OR: 观测到该点的关键帧太少了

* 生成MapPoints

  运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints。防止MapPoints越来越少。然后这一步会根据匹配点增加些新的MapPoints，双目之间可以通过反投影计算，单目可以通过三角化计算。

* MapPoints融合

  检查当前关键帧与相邻帧(两级相邻)重复的MapPoints.  也就是说，当前帧中的图像点已经与另一个MapPoint关联起来了，而现在又要与另一个MapPoint进行关联，则此时就会更新(融合)为新的MapPoint进行关联。

  上述过程其实是一个循环过程，处理链表中的关键帧。

* LocalBA

  当队列为空时，就会进行一次局部BA优化。和当前关键帧相连的关键帧即MapPoints做局部BA优化。

* 关键帧剔除

  其90%以上的MapPoints能被其它共视帧(至少3个)观测到的关键帧。 

* 将剩余的关键帧送入到Local Closing中

## Local Closing线程(闭环检测)

该线程的输入数据是来自Local Mapping线程的nlploopKeyFrameQueue.

* 队列中取一帧

* 判断距离上一次闭环检测是否超过10帧，只有超过10帧才会进行闭环检测

* 计算当前帧与相连关键帧的BoW最低得分， BoW得分记为H.BoW

  自适应计算一个阈值。最低得分是minscore。

* **检测得到闭环候选帧**

* 检测候选帧连续性， 通过三层循环实现。**这一部分需要进一步研究。**

### 检测得到闭环候选帧

这个模块的输入是pKF，即当前输入的关键帧；minscore，即计算得到的与关键帧的最低的BoW得分。

* 找出与当前帧有公共单词(BoW)但不包括与当前帧相邻的关键帧，作为候选帧，这里貌似只要有重复的单词存在就行
* 统计候选帧中与当前pKF具有共同单词最多的单词数，记作maxcommonwords。
* 得到阈值： mincommon = 0.8 * maxcommonwords，也就是降低标准了，也就是会有多个关键帧满足这个条件，而不是仅仅就这个提供maxcommonwords的帧才行
* 筛选共有单词大于mincommons, 且H.BoW得分大于minscore的关键帧。将结果保存在`lscoreAndMatch`中，即一个list
* 将存在直接相连的关键帧分为一组，计算组内最高得分`bestAccScore`，表示相连关键帧的个数，同时得到每组中得分最高的关键帧。计算结果保存在`lsAccScoreAndMatch`和`bestAccScore`。
* 计算阈值minScoreToRetain = 0.75 * bestAccScore
* 根据minScoreToRetain与lsAccScoreAndMatch对关键帧筛选，得到vploopCandidates，是一个vector，里面也可能存在多个关键帧，所以在LocalClosing线程的最后一步是对这个向量里面的数据进行连续帧检测。

### Sim3计算

 这一部分没听明白。

## ORBmatcher

* 使用各种技巧减少搜索范围。
* 特征点通过描述子匹配后会进行旋转一致性检测。并且最佳匹配特征点要明显优于次优匹配点。
* 特征点提取仍然是非常耗时的地方。

## Optimizer.cpp





































