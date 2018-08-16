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
  
  * 上述两种距离的加权和(默认)
  
* calculate_scores.py

  主要用于计算分类正确的样本的scores,也就是最后的一层Softmax的输入
  
* OpenMaxLayer.py

  里面是OpenMax算法的主要实现，包括Weibull distribution的拟合等.

* train_om.py

  用于训练ResNet3D模型。

* test_openmax.py

  对OpenMax层的输出进行测试对比其与SoftMax的输出之间的差异。

* test_resnet3d.py

  对ResNet3D模型的输出进行测试。

* showDetectionResults.py

  包含用于显示的函数。
  
* test_openmax.py

  主要用于测试OpenMax的功能，调节参数等
  
* readH5.py & readMat.py

  用于读取.h5数据文件，并生成Datasets,后续使用`gluon.data.DataLoader` 生成DataIter。

  后者用于读取.mat数据文件，并按照9*9的窗口大小将数据分割成子窗口，并以字典的形式返回分割结果。字典包含三个Items:'Datas', 'Rows', 'Cols', 'Boxes'分别对应分割后的9*9*200的子块，子块的行数，子块的列数，每个子块的参数，包括中心坐标、长、宽。

* unknown_top.py

  顶层模块，调用所有的子模块进行未知目标检测。
  
## 依赖

* MXNet, Gluon

  安装教程：[Install MXNet](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU)

* libmr

  文件夹内的`libmr.so`。

* joblib

```Python
    pip install joblib
```

## 使用说明

### 训练ResNet3D模型

可能需要针对自己的数据集进行修改。

* 参数说明

可选参数   |    含义     |   默认值
 ------ | ------ | ------
filename | 训练集文件(.mat格式)  | 'I9-9.h5'(Indian Pines Dataset)
is_train | 指定是否为训练过程     | True
batch_size | 指定训练的batch大小 | 16
epoches  | 指定训练的迭代次数    | 12

* 命令

```Python
    python train_om.py --filename=.. --is_train=.. --batch_size=.. --epoches=..
```

### 计算用于OpenMax的数据

* 涉及到的文件

  * calculate_mavs.py

  * calculate_dists.py

* 命令

```Python
    python calculate_mavs.py 
    python calculate_dists.py 
```

* 说明：

  运行上述命令会保存两个文件，分别为：`mavs.joblib`, `dists.joblib`。

### 测试OpenMax和SoftMax (顶层文件)

* 参数说明

    可选参数   |    说明    |     默认值
    ------   |  ------    |  ------
  img_file   | 指定测试数据的名字        | 'Indian_pines_corrected.mat' (完整的Indian Pines Datasets)
  cls_num    | 在训练过程中已知的类别数   | 9
  net_params | ResNet3D Model的参数    | ResNet3D.params
  ctx        | 运行的Context           | mx.cpu()
  width      | 子窗口的宽               | 9
  height     | 子窗口的高               | 9
  mavs       | 指定上一步中保存的mavs的文件名   | `mavs.joblib`
  dists      | 指定上一步中保存的dists的文件名  | `dists.joblib`
  show_om    | 指定是否显示OpenMax的检测结果   | False
  show_sm    | 指定是否显示Softmax的检测结果   | False
  show_both  | 指定是否显示上述两个的检测结果   | True

* 命令：

```Python
    python unknown_top.py 参数选项
```

* 运行说明

  在运行`unknown_top.py`过程中，函数调用如下：

  * 调用readMat(mat_file_name)读入高光谱数据，格式为(W, H, 200)，最后一个数字是频谱通道数

  * 加载`joblib`文件

  * 调用`generate_child_window`产生子块，这一步后面需要替换成候选框生成函数

  * 调用`detection_om_sm`进行识别检测

  * 调用`showDetectionResults`显示结果
