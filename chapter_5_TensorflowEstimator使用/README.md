#TensorFlowEstimator使用
***
## 目录
- 1 tf_keras_to_estimator.ipynb
- 2 tf_premade_estimators.ipynb
- 3 tf_premade_estimators-new_feature.ipynb
***
### Estimator在TensorFlow中的定位
- TensorFlow原生的High Level API
- 而Keras其实是借来的
- 易于扩展，易于分布式训练 
### 个人认为的TensorFlow三大件
- tf.data
- Estimator与feature columns
- SavedModel(Serving,Lite,JS)
### 如何使用Estimator
1、定义Input函数
2、定义feature columns
3、构建Estimator
4、使用Estimator
### 定义input函数要求如下：
- 返回一个tf.data.Dataset对象
- 该Dataset对象的元素是feature和label的tuple
- feature是一个python Dictionary
- key是字符串，代表Feature的名称
- Value是Numpy数组或tf.Tensor
- label直接是Numpy数组或tf.Tensor
- input函数是无参数函数，可使用闭包将其参数外置
### 如何理解feature columns
- 是原始数据与训练数据之间的桥梁
- 解决的问题：模型如何解读传递过来的数据
- 问题根源：模型只能处理数字
- 本质上，是特征工程，数据预处理的过程
### 如何使用feature columns
- 在Estimator实例化时作为一个参数
- 是一个list
- list中每个元素都必须是tf.feature_column的一个对象
- 顺序不重要，数据通过key建立对应关系
### feature columns的使用流程
1. 列出input函数中feature全部的key
2. 分析每个key对应的原始数据，确定如何预处理
3. 在tf.feature_column中找到对应的预处理方式（后续也可以做特征工程）
4. 建立空list，将每个key对应的数据按照对应预处理方式处理后添加到list中
5. 在Estimator实例化的时候将此list作为参数传入
### 通过理论知识分析自己的机器学习任务
    适合哪个Estimator
    适合什么样的超参数
    如何做特征工程  
**训练**     
- classifier.train(input_fn=lambda:input_fn(train, trian_y, training=True), steps=5000)  
**要点**  
- 有了Estimator后，训练只需要指定input函数
- input函数必须是无参数函数，这里通过匿名函数侧面实现了闭包解决了此问题
- Dataset如果是有限的，那么到头后自动停止训练
- 如果是无限的，需要制定steps
### 总结
- 处理机器学习项目，Estimator是个不错的选择
- 处理深度学习项目，暂时优先考虑Keras
- Data Pipeline + High Level API + SavedModel是大势所趋
