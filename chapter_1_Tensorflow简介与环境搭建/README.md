# TensorFLow简介与环境搭建
***
### 回顾一下什么是深度学习
- 出发点：数据+任务目标
- 有了任务目标，数据也就可以分成：有价值的数据+噪声
- 这里的噪声是相对于任务目标而言的
- 那么深度学习的每一个Layer，就是用来放大前者，抑制后者的
- 经过一系列这样的Layer后，任务目标突然变的很简单
- 这就是深度学习
### 深度学习计算的本质
- 深度学习计算的本质是张量（tensor)的计算
- 将数据进行一系列的几何变换，使得复杂问题简单化
- 深度学习的训练过程就是这一系列几何变换的搜索过程
- 举例说明：两张不同颜色的纸揉成了一个纸团，希望将两张纸分开
### 神经网络的基本要素
- Layer与Model：一系列的层构成了模型
- Input Data：决定了任务目标的上限
- Loss：决定了训练的反馈信号
- Optimizor：决定了怎么利用反馈信号去调整模型
### 层
- 神经网络的基本数据结构
- 是一个数据处理模块
- 将一个或多个Tensor转换为一个或多个Tensor
- 大部分层是有状态的，小部分是无状态的
- 有状态的层，其状态的表示就是层的权重
- 权重，就是通过Loss和Optimizor学习到的知识
### 模型
- 模型是一系列层连接而成的有向无环图
- 模型的拓扑结构，决定了你的搜索空间
- 搜索空间过大，你可能找不到里面最合适的数据表示
- 搜索空间过小，你的数据表示的表达效果是有上限的
### Loss函数
- 在模型定义好的搜索空间内前进的引路人
- 正确的目标，是第一要务，目标有问题，执行力越强死得越快
- 常见的问题不是Loss函数与任务不相关，而是不那么相关
- Loss的副作用会在某一刻毁掉你前面的一切努力
### 什么是Tensorboard
**官方说明**
- Multiple dashboards(scalars,graph,histograms,images,etc)
- slice and dice data by run and tag
- see how data changes over time    

**从界面说起（DashBoard)**
- Scalars
- Graph
- Distributions
- Histograms
- Image
- Audio
- Text
- Embedding Projector

**从工程师角度看**  
- 交互式数据可视化的Web前端应用
- Tensorboard有自己的后端服务接口
- 自动读取日志数据
- 日志从哪里来？
### Tensorboard的数据来源：tf.summary
- tf.summary本质上也是一个op
- 与普通的op区别在于，输出的结果写入到了硬盘上
- Tensorboard会按照符合一定负责的目录结构去读取日志文件夹里的数据
- 按照一定的时间间隔，重新加载日志文件夹中的数据
- Tensorboard与模型训练之间是解耦的，只要有日志文件夹就可以使用TensorBoard
### 怎么用TensorBoard
- 思考哪些信息最有价值
- 思考如何得到这些信息
- 将这些信息用Tensor的方式计算
- 规划日志文件位置和结构
- 通过tf.summary写入到硬盘
- 启动模型训练，启动TensorBoard服务
### 最合适Tensorboard的应用场景：Jupyter NoteBook
- 需要安装jupyter-tensorboard库 （pip install jupyter-tensorboard)
- 需要使用Magic Method：load_ext加载插件 (%load_ext tensorboard)
- % trensorboard --logdir logs  
![tensorboard](https://github.com/XuLongjia/Tensorflow2.0Learning/blob/master/images/tensorboard.png)