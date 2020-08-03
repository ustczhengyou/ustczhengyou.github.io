### 背景

一般在特征进行embedding之后，接下来的步骤是如何对特征进行组合形成交叉组合特征，来增加模型表达能力。对于DNN来讲虽然能够学习到特征之间的非常复杂交叉关系，但是相比于cross network需要指数级别的参数，不能显示地形成特征交叉能力，并且不一定能够有效地学习一些特征交叉的形式。

`DNN has the promise to capture very complex interactions across features; however, compared to our cross network it requires nearly an order of magnitude more parameters, is unable to form cross features explicitly, and may fail to efficiently learn some types of feature interactions. Deep neural networks (DNN) are able to learn non-trivial high-degree feature interactions due to embedding vectors and nonlinear activation functions.`

所以DCN模型的具有能够显示地并更有效地学习一些特征交叉项的能力。[特征多项式之间的显示交叉]

`that is more efficient in learning certain bounded-degree feature interactions`

### 模型结构

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200520113648361.png" alt="image-20200520113648361" style="zoom:60%;" />

- **Cross Layer**

  <img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200520113732453.png" style="zoom:60%;" />

- **Cross Layer分析**

  - polynomial approximation

    逼近理论：任何函数在一定的平滑度假设下都可以通过多项式来任意精度地逼近。

    `By the Weierstrass approximation theorem , any function under certain smoothness assumption can be approximated by a poly-nomial to an arbitrary accuracy.`

  - generalization to FMs

    

  - efficient projection