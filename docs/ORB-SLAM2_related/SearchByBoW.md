# Search By BoW

smh

2018.11.26

*The Devil is In The Detail.*

## 简介

SearchByBow主要完成两帧之间MapPoint的跟踪。函数重载了两种情况：

* 当前帧与关键帧
* 两个关键帧之间 

后者貌似通常用于回环检测。因此这里主要对第一种情况进行说明。这个函数主要用于完成对**关键帧**中MapPoint的跟踪。函数原型为:

```c++
int ORBmatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches);
```



## 实现原理

函数需要完成两个功能，(1) 在当前帧中找到与关键帧的特征点相匹配的特征点，然后设置其对应的MapPoint为关键帧中的匹配特征点的MapPoint；(2) 根据旋转一致性，对上述计算得到的MapPoint进行检测，剔除不好的点。

### 特征点匹配

1. 首先获取关键帧的MapPoints

   ```c++
   const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
   ```

2.  然后是三层循环

   1.  先遍历关键帧中的每一个Node，只有属于同一Node，两个特征点才有可能是匹配的
   2.  然后遍历该Node中包含的所有关键帧的特征点
   3.  然后遍历当前帧的所有特征点，并计算与关键帧特征点的距离，计算最小的两个距离及特征点

   这里有一个trick，计算出来两个表现最好的特征点后，最近的距离要在一定程度上小于次近的距离，此时才认为最近匹配是比较靠谱的。在第三步中，还会统计所有特征点的方向相对于关键帧中相匹配特征点的方向的变换梯度直方图。

### 旋转一致性

​     在完成上面的三层遍历后，进行旋转一致性检测，即相互匹配的特征点之间的旋转角度差应该在整幅图像中相关一致。具体实现就是提出三个旋转角度数量最多的其它所有特征点，将这些点的MapPoint重新设置为NULL。

### TODO

这里还没有搞懂BoW中的Node到底是指什么？

### 实现代码

略。