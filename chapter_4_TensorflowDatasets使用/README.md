#TensorFlowDatasets使用
***
##目录
- 1 tf-data_basic_api.ipynb
- 2 tf_data_generate_csv.ipynb
- 3 tf-tfrecord_basic_api.ipynb
- 4 tf_data_generate_tfrecord.ipynb
***
### 为什么需要tf.data
**回顾一下整个机器学习项目的数据流**  
1. 拿到未清洗的原始数据集
2. 清洗原始数据集得到干净的留存数据集
3. 将留存数据集处理成模型需要的输入数据集
4. 训练模型，导出训练好的模型
5. 将训练好的模型封装成API
6. 将真实的原始数据处理成可以输入模型API的数据
7. 送进模型API，得到模型处理的结果
8. 将模型输出的结果加以处理，变成可读性强的最终数据结果 
 
**回顾一下将硬盘上的数据送到GPU上训练的过程** 
- GPU将硬盘上的数据加载到内存上，形成一个Batch
- GPU启动并预热，建立模型，初始化模型参数（仅首次）
- CPU将内存上的一个batch数据送到GPU上，执行训练  

**迫切需要一个能将硬件资源利用率最大化的工具**  
- 加载数据无需等待GPU信号
- 模型训练无需等待CPU的IO操作
- 能方便安全的实现多线程加速
- 能简单快捷的实现常用的数据集的操作
- 从原始数据到模型需要的数据之间的过程变成可复用的代码  
### 怎么使用tf.data
### 数据管道三步走
1. from 数据源 to Dataset
2. from Dataset to Dataset
3. fromDataset to Iterator
### Dataset是什么
- dataset是一个可迭代的Python对象
- dataset想是一个“结构体数组”
- 这个“结构体”里面的每一项都是Tensor
- 每一个这个Tensor都有一个tf.DType和一个tf.TensorShape
- Dataset直接在自己身上调用方法便可以修改自身
### 原始数据变为Dataset
- 大部分常见数据格式都可以直接变为Dataset类型
- 也可以先通过Numpy和Pandas转化为Tensor
- 小数据集建议全部加载到内存里处理
- 大数据集先转化为TFRecord，再搭建Dataset管道
- 复杂情形，可以通过Generator的方式构建Dataset
### Dataset的常见处理函数
- batch
- cache
- prefetch
- range
- concatenate
- filter
- from_generator
- from_tensor_slices
- interleave
- map
- repeat
- shard
- shuffle
- window
- zip
### Dataset处理完毕后如何喂给Model
- 在使用High-Level API时，可以直接用作Input
- 在使用Low-Level API时，__iter__
### 总结
- tf.data的核心是：Dataset
-  Dataset的本质是个结构体数据
- 把Dataset当成一个普通的数组就好了