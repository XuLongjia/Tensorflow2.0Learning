#TensorFlow基础API使用
***
##目录
- 1 tf_basic_api.ipynb
- 2 tf_keras_regression-customized_loss.ipynb
- 3 tf_keras_regression-customized_layer.ipynb
- 4 tf_function_and_auto_graph.ipynb
- 5 tf_diffs.ipynb
- 6 tf_keras_regression-manu-diffs.ipynb 
***
### 如何理解TensorFlow Tensor
- Tensor的本质是一套计算流程
- 至于计算什么时候执行，在哪执行，另说
- 在Eager Execution之前，Tensor都是只定义，不计算的
- 在Eager Exectution之后，Tensor定义好后，结果也计算好了
### Numpy ndarray VS tf.Tensor VS tf.Variable
- 逐级封装，增加功能
- tf.Tensor不可更改
- Numpy ndarray不可放在GPU或者TPU上
- 若tf.Tensor放在CPU上，在执行tf.Tensor.numpy()时可以共享内存
- tf.Variable是面向用户的，而tf.Tensor是面向TensorFlow自己的
### GradientTape是什么
- 从python角度看：上下文管理器
- 从TensorFlow角度看：梯度追踪器
- 从使用角度看：可以一次性，也可以重复使用
- 从追踪对象角度看：默认追踪Variable，其余需要显性watc