# ORB-SLAM2中的三种Track模型

smh

2018.11.26

*The Devel is in the detail.*

## 三种Track 模型

* Track With Motion Model
* Track Reference Keyframe
* Relocalization 

### 共同的思想

1.  首先计算当前帧的初始位姿，比如分别使用恒速模型、把上一帧的位姿当做当前帧的位姿
2.  计算当前帧的地图点，比如使用初始位姿、BoW模型等
3.  使用计算得到的地图点对位姿进行优化
4.  剔除外点

## Track With Motion Model

使用Search By Projection完成对地图点的跟踪。

## Track Reference Keyframe

使用Search by BoW完成对地图点的跟踪。大概流程如下：

1.  计算当前帧的BoW
2.  调用SearchByBow计算当前帧的MapPoints，保存在`vpMapPointMatches`，并计算匹配的点数
3.  以上一帧的Pose初始化当前帧的Pose
4.  调用PoseOptimization对上一步的位姿初始值进行优化
5.  根据优化后的位姿，更新SearchByBoW中计算得到的MapPoints
6.  返回`nmatchesMap >= 10`

