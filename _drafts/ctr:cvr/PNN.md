### 模型结构

##### 总体结构

<img src="/Users/youzheng/Library/Application Support/typora-user-images/image-20200515151107119.png" alt="image-20200515151107119" style="zoom:60%;" />

##### 核心Layer

- Inner-Product
  - 
- Outer-Product

### 特征类型

存在 Multi-fields Categorical的离散类型特征。这类特征如果用one-hot进行编码会存在高维稀疏的特点，很容易过拟合。同时field项之间存在着依赖和层级关系，采用何种结构来捕捉这种关系呢？

Thus we are seeking a DNN model to **capture high-order latent patterns** in multi-field categorical data. And we come up with the idea of product layers to **explore feature interactions automatically**. 

最主要目的是模型能够自动探索交互特征之间的关系，通过类似FM的Product Layer层获取特征的一阶交互关系，同时MLP层来捕捉更高阶的关系。









