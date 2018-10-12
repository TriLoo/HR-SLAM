# Attention Sequeeze and Excitation Network


## License

GPL-v2

## Description

This folder includes the implementation of my own network to realize instance-level segmentation

## References

[1] Seqeeze-and-Excitation Network

[2] Spatial Transformer Network

[3] Mask R-CNN

...

## Logs

2018.06.28

决定先基于<RedNet: Residual Encoder-Decoder Network for indoor RGB-D Semantic Segmentation>

开发基础网络结构。现在决定的是，引入STN, 修改目标函数，引入Instance-level segmentation(还不确定怎么做)。

基础网络基于ResNet-50来实现, 后续考虑添加旁路。

2018.07.02

这两天看了几篇论文。主要分为early fusion, late fusion, middle fusion之类的，但貌似Multi-scale Fusion

还是最好的，所以网络结构应该具有多尺度feature fusion的特点；其次，加入我的想法，就是引入Attention

机制。

现在的损失函数都是基于Cross Entropy来做的，这样肯定是不能实现Instance-level Segmentation。

中间<Learning common and specific features ...> 这篇文章的目标函数比较有新意。

下一步怎么做？

2018.08.11

整个实验需要完成的工作：

* 数据集：NYUv2, SUN 3D

* RGB-D (2.5D) Feature Fusion

  * Multiscale (Early-Middle-Late) Feature Fusion

  * Use depth data as 1-channel

  * RDFNet

  * FuseNet

  * RedNet

  * ...

* Attention:

  * Attention ResNet

  * Convoluton Attention Block

* Instance-Level Segmentation

  * Mask R-CNN

  * Learning to segment everything

  * PANet

  * FCIS

  * DeepMask

  * SDS

  * ...

* Target function

  * 还不知道怎么做？


2018.08.16

今天刚把一直有问题的 CSVIter 解决了一下，现在貌似可以工作了。下一步就是先训练像素级的吧。最近重新看了几篇实例分割的论文，感觉比较复杂，通常的做法
是分解成: Object Detection + Semantic Segmentation。这样的话，我还得考虑一下加上Object Proposal的分支，工程量陡增了一下。

2018.09.14

今天算是吧训练部分的代码跑起来了，但误差增加是什么意思？下一步就是用一个1060试一下结果吧。

2018.09.20

用1060跑的后，总是有问题，cudaMalloc错误，感觉是显存不够用。到今天2018.09.27，还是没解决，GPU真是个问题，CPU跑起来后太慢了。

试了Kaggle kernel, google colab等，都没用起来。天池的notebook听说支持gpu了，但申请需要粮票+每次最多只有6个小时。上传数据集也太麻烦了。


