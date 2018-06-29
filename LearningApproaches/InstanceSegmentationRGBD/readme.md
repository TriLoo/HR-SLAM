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
