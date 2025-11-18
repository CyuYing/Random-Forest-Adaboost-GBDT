


















# Random Forest、Adaboost、GBDT

## 一、集成学习

将多个弱监督模型组合，以产生更好的效果（帮助集成模型降低预测的偏差或者方差）。常见的集成学习框架有三种：Bagging，Boosting 和 Stacking
**注：每一个子模型都可以用决策树训练**

<br>


### 1.1Bagging

并联

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd4c843170964cfbb699d978ed0c2e22.png#pic_center)

将训练集有放回的随机抽样，产生多个子训练集，分别训练模型，综合考虑结果
<br>



### 1.2Boosting

级联

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06ab111638c44e57a4c307330d48da3e.png#pic_center)

不随机抽样直接用整个训练集，不断迭代子模型
<br>



### 1.3 Stacking

相比bagging模型

- 允许使用不同类型的模型作为base model
- 使用一个机器学习模型把所有base model的输出汇总起来（加权求和），形成最终的输出

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31178034eaa440118e3a786249c3875b.png#pic_center =80%x80%)
<br>


### 1.4 残差

评价一个模型的预测能力，一般考察残差的两个方面

- 偏差，即与真实值分布的偏差大小
- 方差，体现模型预测能力的稳定性

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d9cbc5b1820f4fabbb172e2237f00a45.png#pic_center =50%x50%)
<br>

<br>


## 二、随机森林

bagging类型，弱分类器

**随机：** 随机抽样

**森林：** 随机样本训练出的多个决策树组合

<br>

### 2.1训练步骤：

1. 分几棵树，每棵树分几层

2. 随机采样，训练每个决策树

   DATASET[N * D] => data subset[n * d]

   N,n 样本数量，N > n

   D,d 特征数量，D > d

3. 将测试集样本输入到每棵树中，再将每棵树结果整合

   Regression：均值

   Classification：众数
   <br>

### 2.2优缺点：

**pros：**

- 随机性强，具有很强的抗噪性（可以避免异常数据的影响）
- 并联同时训练 +随机抽样成小样本，处理高维数据更快
- 树状结构，模型解释性强，每个特征的重要性

**cons：**

- 不具备处理困难样本的能力，由于是弱分类器，导致每一个分类器都无法处理困难样本
  <br>
  <br>

## 三、Adaboost

boosting类型，弱分类器

### 3.1 训练步骤

1. 初始化每一个样本点困难度为1

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f895a43411043ca99f40043e02b678e.png#pic_center#pic_center =30%x30%)
   <br>

2. 利用当前若分类器的训练结果，更新所有样本的困难度

   提高错误样本的困难度，正确样本困难度不变

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/09aeb8836b2c4c5f862b5f313fa0d3eb.png#pic_center =30%x30%)

   正确/错误 = 7/3，将每一个错误点放大三分之七，与正确总权重齐平

   <br>

3. 基于当前困难度，训练下一个分类器

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc3b589f78b94e5aad894dfad7055b72.png#pic_center =30%x30%)

   可以看到，上一轮错误的因为被重点关照（提高权重），使得正确分类

   正确/错误 = 11/3，将每一个错误点放大三分十一，与正确总权重齐平

   

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/401193c682174ac685bfc2bdcbf5c4fe.png#pic_center =30%x30%)

   正确/错误 = 19/3，将每一个错误点放大三分十一，与正确总权重齐平

   <br>

4. 训练结束后，根据每个分类器的正确率，分配对应的权重

   **注意**：我们不能直接使用正确/错误来模拟正确率

   ​	假设有一个学习器的比值是正确/错误 = 19/3，对它的权重与正确/错误 = 19/3的权重并不对称

  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2c66794a1e8b442f9cacd1e774a6aae1.png#pic_center =50%x50%)

   正确做法：取对数作为权重

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c2b5c7de7a014e248f651d5c4078479a.png#pic_center =50%x50%)

   <br>




5. 按权重将每个学习器组合在一起

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af6dde4973484d27b5dff66ab66cbaf2.png#pic_center =30%x30%)

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27199973e9eb41fcbdc5318512534259.png#pic_center =30%x30%)
<br>

### 3.2优缺点

**pros：**

- 级联结构不断迭代 + 利用权重组合结果，可以处理困难问题

**cons：**

- 专注于处理困难问题，会导致对异常数据过于敏感
- 级联训练，前一个训练好才能训练下一个，训练速度慢
  <br>
  <br>

## 四、Gradient Boosting

boosting类型，与Adaboost 的区别在于迭代模型的方法

- 通过调整错分数据点的权重来改进模型
- 通过计算负梯度来改进模型（向极小值逼近的方向）

****

**残差**：真实值与预测值的差

上一轮预测结果的残差作为当前的训练数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3960cab33ddd4f20a4b1a4034cbe997f.png)

****

**梯度下降：** 对目标进行极小化，直到收敛

由于负梯度方向是函数值下降最快的方向，故每一步以负梯度方向更新。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/035989be1de846eb9805ed3004f35d70.png#pic_center =30%x30%)

****



**回归：** 均方差损失函数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36202ce5623b4a34934fcdcc888ae801.png# =20%x20%)

负梯度计算![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0576ffe566054965a411ad9079272a23.png# =20%x20%)，结果正好是残差，故用负梯度近似模拟残差

**原理是残差，但实际算法中是不断及计算梯度来更新模型的**



**分类：** 交叉熵损失函数

- 二分类：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/90516b4cc16742ad8d136a72ff015dc1.png#pic_center =40%x40%)
- 多分类：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4a84235eb964e8591654fa86d8d040b.png#pic_center =24%x24%)

**注：** 一般分类模型给出的都是每个类别的概率，而不是确切的类别



### 4.1 训练步骤

1. 初始化弱学习器，将样本标签的均值做为预测值，得到初始学习器
2. 迭代每一棵树
   1. 对于每个样本计算负梯度（残差）
   2. 将上一步计算的残差作为样本的新真实值
   3. 训练第2棵树来拟合残差（cart回归树）
   4. 用残差更新强学习器，设置学习率，计算损失函数的值
3. 重复，直到无法优化
4. 将每一个学习器的结果相加

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97fb79f27c1f49a3950b606ecb514392.png#pic_center =70%x70%)

### 4.2[复杂的例子](https://zhuanlan.zhihu.com/p/280222403)

### 4.3 优缺点

**pros：**

- 非常灵活，可选的损失函数非常多，并且可以处理连续或离散的数据
- 相对adaboost，异常样本权重不易被提升，对异常值不敏感

**cons（boosting限制）:**

- 串行生成，速度慢
- 数据维度高，计算复杂度高


## 参考文章

[【机器学习】决策树（中）——Random Forest、Adaboost、GBDT （非常详细）](https://zhuanlan.zhihu.com/p/86263786)
[GBDT(梯度提升决策树)——来由、原理和python实现](https://zhuanlan.zhihu.com/p/144855223)
[「五分钟机器学习」集成学习——Ensemble Learning](https://www.bilibili.com/read/cv6765576)
[Manning Editors出版的《Grokking Machine Learning》一书的存储库](https://github.com/luisguiserrano/manning/blob/master/Chapter_12_Ensemble_Methods/Random_forests_and_AdaBoost.ipynb)
[What is AdaBoost? Friendly explanation with code!](https://www.bilibili.com/video/BV1qQkgYTEN4/)



