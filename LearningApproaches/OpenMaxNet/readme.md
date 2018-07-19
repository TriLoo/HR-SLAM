# OpenMaxNet Project

## license

GPLv2

## Description

这个文件夹主要有两个实现：

* OpenMax

    主要参考文献如下：

  * Towards Open Set Deep Networks
  
  * Probability Models for Open Set Recognition
  
  * Toward Open Set Recogniton
  
  * Meta-Recogniton: The Theory and Practice of Recognition Score Analysis
  
* ResNet3D

  这里主要包含具有3D CNN和残差结构特点的网络模型，由于输入的分辨率只有9 * 9，所以分辨率只会降低
  两次。

具体包括以下文件：

* libmr.so

  这个文件主要是用于拟合Weibull Distribution的库，使用时需要在Python中Import进去：
  
  ```Python
  import libmr
  ```
  
* calculate_mavs.py & calculate_dist.py

  这两个文件主要是用于计算Mean Activation Vectors和Scores与MAV之间的距离，共有三种距离计算方式：
  
  * 欧式距离
  
  * 余弦距离
  
  * 上述两种距离的加权和
  
* calculate_scores.py

  主要用于计算分类正确的样本的scores,也就是最后的一层Softmax的输入
  
* OpenMaxLayer.py

  里面是OpenMax算法的主要实现，包括Weibull distribution的拟合等.
  
* test.py

  主要用于测试ResNet3D模型
  
* main.py

  主要用于训练ResNet3D 模型
  
* test_openmax.py

  主要用于测试OpenMax的功能，调节参数等
  
* readH5.py & readMat.py

  用于读取.h5数据文件，并生成Datasets,后续使用`gluon.data.DataLoader` 生成DataIter。

  后者用于读取.mat数据文件，并按照9*9的窗口大小将数据分割成子窗口，并以字典的形式返回分割结果。
  
## 依赖

* MXNet, Gluon

* libmr

* joblib

## 参数

  每个文件的参数是不同的，可以查看每一个.py文件中的`parse`进行确认。
  
## 使用顺序

* 先修改ResNet3D文件，修改网络模型，适合自己的任务

* 然后使用main.py进行训练

* 使用calculate_mavs.py来得到每一类的mavs和scoers并保存成joblib文件

* 使用calculate_dists.py来得到每一类的distance信息

* 使用test_openmax调用OpenMaxLayer来实现Open Set Recognition的功能

  
