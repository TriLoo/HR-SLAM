# RANSAC的实现

## Initializer.cc

在*bool ORB_SLAM2::Initializer::Initializer(...)*中，在单目相机的初始化过程中，使用了RANSAC计算H、F矩阵，然后就用到了RANSAC.

使用RANSAC的目的就是随机选择出8对匹配点(8点法)计算上述两个矩阵，具体的计算算法是DLT。

为了产生8个随机匹配点的索引值，用到了一个二维数组`vector<vector<size_t>>`，数组的行数为最大迭代次数`mMaxIteration`，数组的列数为8，也就是每一行都可以计算一次迭代所需要的八个匹配点(因为需要使用的是八点法)。

产生随机匹配点索引值的代码如下：

```c++
 	// vector<vector<size_t>> mvSets;
	mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    // 用于产生RANSAC的随机索引
    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;  // 每次迭代都是从所有的1 -> N里面选择数字

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            // it是迭代次数，j是八点法中的第j个点
            mvSets[it][j] = idx;

            // 把最后的一个数放在当前这个被使用的索引的位置，防止在最小索引集里面有重复的索引出现了
            // 然后调用pop_back()把最后的一个数删除，最终就实现了产生最小集，也不会使用重复的特征点的情况
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
```

