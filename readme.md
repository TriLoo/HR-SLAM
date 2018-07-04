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

