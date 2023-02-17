# TensorFlow 

## 疑问:whale:
1. 数据集的来源获取？
2. 如何创建模型，模型怎么选择，如何选择层？
~

## 学习感受
> 缺的知识太多了，无法按自己想的输出代码，只能到处抄，很难受~


## Keras
Keras 拥有多种常用的 API，下面是一些常用的 API：
- Sequential：该类实现了一个简单的序列模型，可以通过简单的 .add() 方法添加层。
- Dense：该类实现了全连接层，可以通过它来构建多层感知器模型。
- Conv2D：该类实现了二维卷积层，可以用于构建卷积神经网络（Convolutional Neural Network，CNN）。
- MaxPooling2D：该类实现了二维池化层，可以用于降低图像数据的维度。
- Flatten：该类实现了扁平层，可以将多维数据展开成一维数据。
- LSTM：该类实现了长短时记忆网络（Long Short-Term Memory，LSTM）层，可以用于处理序列数据。
- compile：该函数用于编译模型，在训练前必须调用。
- fit：该函数用于训练模型，可以通过指定训练数据、批次大小和训练轮数等参数来实现训练。
- evaluate：该函数用于评估模型，可以通过指定测试数据来评估模型的准确率。
- predict：该函数用于预测，可以通过输入数据来预测模型


### Keras.Sequential 建模如下：

1. 创建模型：使用 **`model = Sequential()`** 创建一个空的序列模型。
2. 添加层：使用 **`model.add()`** 方法添加各种不同的层。
3. 指定输入形状：需要在第一层指定输入形状，即第一层的 **`input_shape`** 参数。
4. 指定损失函数：在训练模型时，需要指定损失函数，通过 **`model.compile()`** 方法实现。
5. 训练模型：使用 **`model.fit()`** 方法进行训练。
6. 评估模型：使用 **`model.evaluate()`** 方法评估模型的表现。
7. 预测结果：使用 **`model.predict()`** 方法进行预测。

#### tf.feature_column.numeric_column接受参数
- key：特征列的名称，它可以是字符串，数字或者其他类型。
- shape：特征列的维度，可以是整数或者 None，默认值为 None。
- default_value：特征列的默认值，在输入数据中没有提供该特征时使用。
- dtype：特征列的数据类型，默认值为 tf.float32。
- normalizer_fn：对特征数据进行归一化的函数，可以自定义，默认值为 None。

#### tf.keras.layers.Dense
1. **`units`**：该层的输出维度，也就是该层有多少个神经元。
2. **`activation`**：激活函数，默认为 'linear'。常用的激活函数有：ReLU、Sigmoid、Tanh、Softmax 等。
3. **`use_bias`**：是否使用偏置项，默认为 True。
4. **`kernel_initializer`**：权值矩阵的初始化方法，默认为 'glorot_uniform'。
5. **`bias_initializer`**：偏置项的初始化方法，默认为 'zeros'。
6. **`kernel_regularizer`**：权值矩阵的正则化方法，默认为 None。
7. **`bias_regularizer`**：偏置项的正则化方法，默认为 None。
8. **`activity_regularizer`**：激活值的正则化方法，默认为 None。

##### 对activation 中激活函数的类型进行说明
Keras 中的 **`activation`** 参数用于选择激活函数，决定每个神经元的输出。下面是一些常用的激活函数：
1. **`sigmoid`**：Sigmoid 函数是一个常用的二分类激活函数，其输出范围是 [0, 1]。使用 sigmoid 函数通常是预测 0/1 或真/假的问题。
2. **`softmax`**：Softmax 函数是一个常用的多分类激活函数，其输出是每个类别的概率值，并且概率值总和为 1。
3. **`relu`**：ReLU (rectified linear unit) 函数是最常用的激活函数之一，其输出是输入的线性数学变换。该函数定义为 f(x) = max(0,x)，即如果输入值为正，则保留该值，否则将其设置为 0。
4. **`tanh`**：tanh (hyperbolic tangent) 函数是双曲正切函数，其输出范围是 [-1, 1]。它是 Sigmoid 函数的变体，比 Sigmoid 函数更鲁棒。
5. **`linear`**：线性函数，即 f(x) = x。该函数通常用于最后一层，例如回归问题中的输出层。


### Keras 内置了多种损失函数进行解释说明
1. mean_squared_error：平均平方误差。通常用于回归问题。
2. mean_absolute_error：平均绝对误差。
3. categorical_crossentropy：分类交叉熵。通常用于多分类问题。
4. binary_crossentropy：二进制交叉熵。通常用于二分类问题。
5. sparse_categorical_crossentropy：稀疏分类交叉熵。用于多分类问题，但输出的标签是整数类型而不是 one-hot 编码。
6. kullback_leibler_divergence：Kullback-Leibler 分歧。常用于自编码器中。
7. poisson：泊松回归损失。适用于拥有整数输出的回归问题。

## 常见的优化算法说明
- SGD (Stochastic Gradient Descent)：随机梯度下降算法是最简单的优化算法之一，其特点是每次仅使用一个样本来计算损失函数的梯度。SGD 算法的收敛速度较慢，但它简单易用。
- RMSprop：RMSprop 是一种基于 RProp 算法的改进版本，它可以加快收敛速度，解决 SGD 算法的困境。它通过调整学习率来实现这一点。
- Adagrad：Adagrad 算法是一种自适应学习率的优化算法，它具有更快的收敛速度和更好的收敛性。Adagrad 算法通过调整每个参数的学习率来解决问题。
- Adadelta：Adadelta 算法是一种基于 Adagrad 算法的改进版本，它不需要额外的学习率超参数，并且比 Adagrad 算法更加稳定。
- Adam：Adam 算法是一种自适应学习率的优化算法，它结合了基于动量的梯度下降算法和基于 RMSProp 的梯度下降算法的优点，因此它在深度学习领域中非常流行。
- Adamax：Adamax 算法是一种基于 Adam 算法的改进版本，它对学习率进行了优化，更适合

### 评估指标：
- binary_accuracy：二分类准确率
- categorical_accuracy：多分类准确率
- sparse_categorical_accuracy：稀疏多分类准确率
- top_k_categorical_accuracy：多分类前k准确率
- sparse_top_k_categorical_accuracy：稀疏多分类前k准确率
- cosine_proximity：余弦相似性
- mean_absolute_error：平均绝对误差
- mean_squared_error：均方误差
- mean_squared_logarithmic_error：对数均方误差
- hinge：悬挂误差
- squared_hinge：平方悬挂误差
- categorical_crossentropy：多分类对数损失
- sparse_categorical_crossentropy：稀疏多分类对数损失
- binary_crossentropy：二分类对数损失

### 回调函数
ModelCheckpoint：定期保存模型，第二个epoch没有第一个好，就不保存第二个epoch，直接在第一个基础上进行第三个epoch。
EarlyStopping：早停函数，当培训不再改善验证指标时，停止培训。
 ```
monitor: 被监测的数据。
min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
patience: 没有进步的训练轮数，在这之后训练就会被停止。
verbose: 详细信息模式。
mode: {auto, min, max} 其中之一。 在 min 模式中， 当被监测的数据停止下降，训练就会停止；在 max 模式中，当被监测的数据停止上升，训练就会停止；在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
 ```
Tensorboard：后期介绍
CSVLogger：将结果（损失和指标）写在csv数字上面。
 