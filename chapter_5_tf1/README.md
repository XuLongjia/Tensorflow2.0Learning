# TensorFlow1.0
***
- 1 tf1_dense_network.ipynb
- 2 tf1_dataset.ipynb
- 3 tf1_initialized_dataset.ipynb
- 4 tf1_customized_estimator.ipynb

## tf1.0与tf2.0的区别
- tf1.0:Sess、feed_dict、placeholder被移除
- tf1.0: make_one_shot(initializable)_iterator被移除
- tf2.0: eager mode,@tf.function与AutoGraph
- eager mode vs sess
    ```python
    # Tensorflow 1.x
    outputs = session.run(f(placeholder), feeed_dict={placeholder:input})
    # Tensorflow 2.0
    outputs = f(input)
    ```
    - tf.function 与autograph
        - 性能好
        - 可以导入导出为SavedModel
    - Eg:
        - for/while -> tf.while_loop
        - if -> tf.cond
        - for _ in dataset -> datset.reduce
## api变动
- 有些空间层次太深
    - tf.saved_model.signature_constants.CLASSIFY_INPUTS
    - tf.saved_model.CLASSIFY_INPUTS
- 重复API
    - tf.layers -> tf.keras.layers
    - tf.losses -> tf.keras.losses
    - tf.metrics -> tf.keras.metrics
- 有些api有前缀应该建设子空间
    - tf.string_strip -> tf.string.strip
- 重新组织
    - tf.debugging、 tf.dtypes、 tf.io、 tf.quantization等
## 如何升级
- 替换session.run
    - feed_dict、tf.placeholder变成函数调用
- 替换api
    - tf.get_variable替换tf.Variable
    - variable_scope被替换为以下东西的一个：
        - tf.keras.layser.Layer
        - tf.keras.Model
        - tf.Module
- 升级训练流程
    - 使用tf.keras.Model.fit
- 升级数据输入
    - Itertor变成直接输入

如何将tf1代码转换成tf2的代码（demo)   
tf1:
```python
in_a = tf.placeholder(dtype=tf.float32,shape=(2))
in_b = tf.placeholder(dtype=tf.float32,shape=(2))

def forward(x):
    with tf.variable_scope("matmul",reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W",initializer=tf.ones(shape=(2,2),
                            regularizer=tf.contrib.layers.l2_regularizer(0.04)))
        b = tf.get_variable("b",initializer=tf.zeros(shape=(2)))
    return W * x +b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss = tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())
    outs = sess.run([out_a,out_b,reg_loss],feed_dict={in_a:[1,0],in_b:[0,1]})
```
如何升级:
```python
W = tf.Variable(tf.ones(shape=(2,2)),name='W')
b = tf.Variable(tf.ones(shape=(2)),name='b')
@tf.function
def forward(x):
    return W * x + b
out_a = forward([1,0])
print(out_a)
```