# TensorFlowKeras实战
***
## 目录
- 1 tf_keras_classification_model.ipynb
- 2 tf_keras_classification_model-normalize.ipynb
- 3 tf_keras_classification_model-callbacks.ipynb
- 4 tf_keras_regression.ipynb
- 5 tf_keras_classification_model-dnn.ipynb
- 6 tf_keras_classification_model-dnn-bn.ipynb
- 7 tf_keras_classification_model-dnn-selu.ipynb
- 8 tf_keras_classification_model-dnn-selu-dropout.ipynb
- 9 tf_keras_regression-wide_deep.ipynb
- 10 tf_keras_regression-wide_deep-subclass.ipynb
- 11 tf_keras_regression-wide_deep-multi-input.ipynb
- 12 tf_keras_regression-wide_deep-multi-output.ipynb
- 13 tf_keras_regression-hp-search.ipynb
- 14 tf_keras_regression-hp-search-sklearn.ipynb
***
### tf.keras VS Keras
- keras是一个易上手的通用学习框架
- keras可以选择不同的后端计算引擎
- 既可以通过TensorFlow安装Keras
- 也可以通过Keras安装TensorFlow
- tf.keras针对TensorFlow做了大量优化和扩展
- 不推荐Keras with TensorFlow Backend
### keras工作流
1. 导入数据
2. 数据处理
3. 模型定义
4. 模型编译
5. 模型训练
6. 模型导出
7. 模型应用
### 需要学习的keras模块
- Dataset
- Model
- Layer
- Loss
- Optimizer
- Metrics
### 什么是Keras Callback？
- Callback是一组函数对象
- 在训练过程中的特定时期被调用执行
- 这些函数对象可以在训练过程中访问，保存，修改训练中的参数
- 相当于在训练之前写好了几个锦囊，在特定时期打开并执行
### 为什么需要Keras Callback？
- 核心问题：解决训练启动后的失控问题
- 我们不知道训练多少轮可以得到想要的结果
- 我们不知道什么时候开始模型已经开始过拟合
- 我们不希望模型启动训练后，就没事过去看一眼
- 有时候我们希望训练到一定程度后，记载另一组权重继续训练
### 如何使用Keras Calback？
- Callback需要先实例化
- 实例化之后，以list形式传给model.fit()的callback参数
### Keras内置的Callback
- 动态模型保存：ModelCheckpoint
- 动态训练终止：EarlyStopping
- 远程事件监控：RemoteMonitor
- 自定义动态学习率：LearningRateScheduler
- 数据可视化：TensorBoard
- Plateau策略学习率：ReduceLROnPlateau
- CSV格式训练日志持久化：CSVLogger
### 什么时候要想起Callback？
- 希望在启动训练只有按照一定的策略去更新部分参数
- 希望全面只管的监控训练过程
- 希望在远程了解训练情况
- 希望按照一定的条件自动保存模型
- 希望按照一定的条件自定停止训练
- 希望按照一定的条件自动重新加载特定权重
### Callback的本质
- 将训练过程中可以传递出来的内在信息最大化
- 将训练过程中的失控感降到最低
- 实现定制化的训练流程
### 什么时候使用Functional API构建Model？
- Multi-Input
- Multi-Output
- Shared Layer
- Building Graph of Layers,not Stack
- 用习惯了以后，建议一直使用
### Functional API有什么特点？
- 始于Input Layer
- 终于Model
- 中间用各种Layer把各个Tensor连接起来
- 每一个Layer的Class第一次调用返回的是Layer的Instance
- Layer的Instance可以继续在Tensor上调用
- Tensor的变量名是可以重复使用的
### Layer
### 从面向对象角度理解Keras Layer
- Layer的文档不在TensorFlow官网，而在Keras官网
- 每一个Layer都是一个可调用的函数
- 每一个Layer被调用后返回的都带有通用的Layer接口
- 每一个Layer Instance也是可以在Tensor上面调用的
- 每一个Layer关注的重点，是他们的Input和Output
- 每一个Layer关注的细节，是他们的主要参数是什么
### 何时需要自定义一个Layer
- 给已有的Layer添加特殊功能
- 实现大型网络时将部分Layer模块化
- 实现全新的Layer
### 如何分析Layer
- Layer创建时，需要哪些参数
- Layer训练时，是否有可训练的参数来保存状态
- Layer从Input到Output的计算过程具体是什么
- 给定Layer的Input的shape，是否能推导出Output的shape
### Layer基类的内在执行流程
1. 编写代码，定义NewLayer
2. 在程序中通过NewLayer类实例化一个newlayer实体，此时执行__init__()
3. 在构建完整个Model后，启动训练，此时开始不断重复执行call()
4. 但如果是第一次执行call()，则自动先调用一次build()  

**执行情况总结：**
- \_\_init__()：实例化时执行一次
- build()：在执行call()之前执行一次
- call()：不断的重复执行  

**编写惯例：**   
- \_\_init__()：基于除shape之外的参数进行初始化（input free)  
- build()：基于input_shape参数，完成其余初始化工作
- call()：基于inputs参数，Forward Computation
### 总结
- TensorFlow为Keras做了极大的赋能，请使用tf.keras
- 学好Keras，可以完成80%的Deep Learning任务
- Keras & TensorFlow是你的剑
- 机器学习与深度学习理论是你的内功