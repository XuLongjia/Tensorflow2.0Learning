# TensorFlowGPU与分布式
***
## 目录
- 1 tf_gpu_1.ipynb
- 2 tf_gpu_2-visible_gpu.ipynb
- 3 tf_gpu_3-virtual_device.ipynb
- 4 tf_gpu_4_manual_multi_gpu.ipynb
- 5 tf_gpu_4-manual_multi_gpu_model.ipynb
- 6 tf_distributed_keras_baseline.ipynb
- 7 tf_distributed_keras.ipynb
    - MirroredStrategy在keras模型上的使用
- 8 tf_distributed_estimator_baseline.ipynb
- 9 tf_distributed_estimator.ipynb
    - MirroredStrategy在estimator模型上的使用
- 10 tf_customized_training_baseline.ipynb
    - MirroredStrategy在自定义训练流程上的使用
- 11 tf_distributed_customized_training.ipynb
***
## GPU设置
- 默认用全部GPU并且内存全部沾满
- 如何不浪费内存和计算资源？
    - 内存自增长
    - 虚拟设备机制
- 多GPU使用
    - 虚拟GPU & 实际GPU
    - 手工设置 & 分布式设置
***
## 相关API
- tf.debugging.set_log_device_placement
    - 在训练的时候打印一些信息:某个变量分配在哪个设备上
- tf.config.set_soft_device_placement
- tf.config.experimental.set_visibel_devices
    - 本进程所见的设备
- tf.config.experimental.list_logical_devices
    - 获取所有的逻辑设备
    - 一个实际设备分成好几个逻辑设备，类似于一个硬盘几个分区
- tf.config.experimental.list_physical_deivices
    - 获取实际设备
- tf.config.experimental. set_memory_growth
    - 内存自增
- tf.config.experimental.VirtualDeviceConfiguration
    - 在实际设备上设计逻辑分区
- tf.config.set_soft_device_placement
    - 自动把某个计算分配到某个设备上去，手动设置容易出错，前提是这个设备支持这个计算
- linux: nvidia-smi 
    - 查看显卡信息
    > fit_generator貌似还不支持分布式，只能用model.fit  
## 分布式策略
- MirroredStrategy
    - 同步式分布式训练
    - 适用于一机多卡情况
    - 每个GPU都有网络结构的所有参数，这些参数会被同步
    - 数据并行
        - batch数据且为N份分给各个GPU
        - 梯度聚合然后更新给各个GPU的参数
- CentralStorageStrategy
    - MirroredStrategy的变种
    - 参数不是在每个GPU上，而是存储在一个设备上
        - CPU或者唯一的GPU上
    - 计算是在所有GPU上并行的
        - 除了更新参数的计算之外
- MultiWorkerMirroredStrategy
    - 类似于MirroredStrategy
    - 适用于多机多卡情况
- TPUStrategy
    - 与MirroredStrategy类似
    - 使用在TPU上的策略
- ParameterServerStrategy
    - 异步分布式
    - 更加适用于多规模分布式系统
    - 机器分为Parameter Server和worker两类
        - parameter server负责整合梯度，更新参数
        - Worker负责计算，训练网络
- distirbuted_subgradient_descent   
![ParameterServerStrategy](https://imgchr.com/i/GL4cq0)  
- 分布式类型-同步-MultiworkerMirroredStrategy  
![MultiworkerMirroredStrategy](https://imgchr.com/i/GL46rq)  
- 分布式类型-异步-ParameterServerStrategy  
![ParameterServerStrategy](https://imgchr.com/i/GL4yMn)  

- 同步和异步的优劣
    - 多机多卡
        - 异步可以避免短板效应
    - 一机多卡
        - 同步可以避免过多的通信
    - 异步的计算会增加模型的泛化能力
        - 异步不是严格正确的，所以模型更容忍错误