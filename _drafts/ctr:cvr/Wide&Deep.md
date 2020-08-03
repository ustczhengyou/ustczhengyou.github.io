#### 背景

对于推荐系统来讲，同时要考虑多样性即给用户带来惊喜的体验；推荐系统和搜索一个很大的不同点在于: 搜索场景下，用户的目的比较明确，倾向于内容的相关性。而推荐则是一个目的不是很明确的场景，推荐的item主要源于两种思路：一个根据用户的历史内容及上下文来推荐相关的item(相关性)，另外一个是需要多样性来发掘用户的其他兴趣点(多样性)。在设计模型的时候我们希望模型能够兼顾这两个能力。用户的偏好需要从历史中去挖掘所以有了Memory，同时从历史中发掘潜在的兴趣点即Generation。在模型结构和特征上：**Memory** 主要是从历史数据中学习item(或者特征)的关联性或者频繁度，所以 "统计类特征+低阶的人工交叉特征" 是其主要部分；而 **Generation** 部分要获取的是Wide缺少的泛化能力，从未出现的关系中挖掘潜在关联点。

**线性模型并不能学习到不同特征之间的关联性，除非是从关联特征构造出发，否则各个特征仍然是独立的。**



#### 模型结构

![image-20200518201728260](/Users/youzheng/Library/Application Support/typora-user-images/image-20200518201728260.png)

实例：

![image-20200518203103732](/Users/youzheng/Library/Application Support/typora-user-images/image-20200518203103732.png)

#### Tricks

- Cross-product  transformation

  $\phi(X)=\Pi_{i=1}^dx_{i}^{c_{ki}}$  $c_{ki} \in\{0,1\}$

  还是需要手动人工构造一些特征。

  **This captures the interactions between the binary features, and adds nonlinearity to the generalized linear model.**

- Continuous real-valued features transformation

  `Continuous real-valued features are normalized to [0, 1] by mapping a feature value x to its cumulative distribution function P (X ≤ x), divided into n_q quantiles. The normalized value is (i−1)/(n_q-1) for values in the i-th quantiles.`

  为什么要将连续特征离散化呢？

  - 离散化后的特征对异常数据有很强的鲁棒性

  - 离散化特征等于增加了模型的非线性能力，例如后续的特征交叉等。

    *模型是使用离散特征还是连续特征，其实是一个“海量离散特征+简单模型” 同 “少量连续特征+复杂模型”的权衡。*

