# basicAPI

#### 数据类型：

- tf.int32 
- tf.float32  
- tf.float64
- tf.bool
- tf.string

#### 创建一个张量tensor

`tf.constant(张量内容，dtype=数据类型)`

```python
import tensorflow as tf
a = tf.constant([1,5],dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)
```

将numpy的数据类型转换成tensor数据类型`tf.convert_to_tensor(数据名，dtype=数据类型)`

```python
import tensorlfow as tf
import numpy as np
a = np.arange(0,5)
b = tf.convert_to_tensor(a,dtype=tf.int64)
print(a)
print(b)  #可以看到numpy格式被转换成tensor格式
```

- 创建全为0的张量  `tf.zeros(维度)`
- 创建全为1的张量  `tf.ones(维度)`
- 创建全为指定值的张量 `tf.fill(维度，指定值)`

```python
a = tf.zeros([2,3])
b = tf.ones(4)
c = tf.fill([2,2],9)
```

生成正态分布的随机数，默认均值为0，标准差为1

```python
tf.random.normal(维度，mean,stddev)
```

生成截断式正态分布的随机数

```python
tf.random.truncated_normal(维度，均值，标准差)
```

在`tf.random.truncated_normal`中如果随机生成的数据的取值在(u-2σ，u+2σ)之外，则重新生成，从而保证了生成值在均值附近

```python
d = tf.random.normal([2,2],mean=0.5,stddev=1)
e = tf.random.truncated_normal([2,2],mean=0.5,stddev=1)
```

生成均匀分布随机数

```python
tf.random.uniform(维度，minval,maxval)
f = tf.random.uniform([2,2],minval=1,maxval=10)
```

#### 常用函数：

- 强制tensor转换为该数据类型：`tf.cast(张量名，dtype=数据类型)`

- 计算张量维度上元素的最小值：`tf.reduce_min(张量名）`
- 计算张量维度上元素的最大值：`tf.reduce_max(张量名)`

```python
x1 = tf.constant([1,2,3],dtype=tf.float64)
print(x1)
x2 = tf.cast(x1,tf.int32)
print(x2)

print(tf.reduce_min(x2))
print(tf.reduce_max(x2))
```

#### 理解axis

在一个二维张量或者数组中，可以通过调整axis等于0或1来控制执行维度

- axis=0表示跨行(经度，down）
- axis=1表示跨列（纬度，across)
- 如果不指定axis，则所有的元素参与运算。

计算张量沿着指定维度的平均值

```python
tf.reduce_mean(张量名，axis=操作轴)
```

计量张量沿着指定维度的和

```python
tf.reduce_sum(张量名，axis=操作轴）
```

```python
x = tf.constant([[1,2,3],[2,2,3])
print(x)
print(tf.reduce_mean(x))
print(tf.reuce_sum(x,axis=1))
```

#### tf.Variable

将变量标记为"可训练"，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数。
`tf.Variable(初始值)`

```python
w = tf.Variable(tf.random.normal([2,2],mean=2,stddev=1))
```

#### Tensorflow中的数学运算

```python
tf.add(张量1，张量2)
tf.substract(张量1,张量2)
tf.multiply(张量1，张量2)
tf.divide(张量1，张量2)
# 对应四则运算（只有维度相同的张量才可以运算）
```

```python
tf.square(张量名)
tf.pow(张量名，n次方数)
tf.sqrt(张量名) # 对应平方、次方与立方
tf.matmul（矩阵1，矩阵2）# 对应矩阵乘法
```

四则运算距离：

```python
a = tf.ones([1,3])
b = tf.fille([1,3],3.)
print(a)
print(b)
print(tf.add(a,b))
print(tf.substract(a,b))
print(tf.multipy(a,b))
print(tf.divide(b,a))
```

#### tf.data.Dataset.from_tensor_slices

切分传入张量的第一维度，生成输入特征/标签对，构建数据集

```python
data = tf.data.Dataset.from_tensor_slices((输入特征，标签))
#注：numpy和tensor格式都可以用该语句读入数
feature = tf.constant([0.89,1.90,1.88,0.45])
labels = tf.constant([0,1,1,0]
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
print(dataset)
for element in dataset:
    print(element)
```

#### tf.GradientTape

```python
# with结构记录计算过程，gradient求出张量的梯度
with tf.GradientTape() as tape:
    # 若干计算过程
grad = tape.gradient(函数，对谁求导）

with tf.GradientTape() as tape:
    w = tf.Variabel(tf.constant(3.0))
    loss = tf.pow(w,2)
grad = tape.gradient(loss,w)
print(grad)
```

#### tf.one_hot

独热编码函数

```python
tf.one_hot(待转换数据，depth=几分类)
classes = 3
labels = tf.constant([1,0,2]) #输入的元素值最小为0，最大为2,
output = tf.one_hot(labels,depth=classes)
print(output)
```

#### tf.nn.softmax

输入张量，计算softmax

```python
y = tf.constant([1.01,2.01,-0.66])
y_pro = tf.nn.softmax(y)
print(y_pro)
```

#### assign_sub

赋值操作，更新参数的值并返回
调用assign_sub前，先用tf.Variable定义变量w为可训练（可自更新）
w.assign_sub(w要自减的内容）

```python
w = tf.Variable(4)
w.assign_sub(1)
print(w)
```

#### tf.argmax

返回张量沿指定维度最大值的索引
`tf.argmax(张量名，axis=操作轴)`

#### irirs数据集

```python
from sklearn import datasets
x_data = dataset.load_iris().data
y_data = dataset.load_iris().target
```

#### tf.where()

类似于python中的三元运算符

```python
import tensorflow as tf
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("c：", c)
```

#### np.random.RandomState.rand()

返回一个[0,1)之间的随机数

```python
np.random.RandomState.rand(维度)
import numpy as np
rdm = np.random.RandomState(seed=1)
a = rdm.rand() #返回随机标量
b = rdm.rand(2,3)#返回一个2行3列的随机数矩阵
```

#### np.vstack()

将两个数组按垂直方向叠加
np.vstack(数组1，数组2)

```python
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack((a,b))
print("c:\n",c)
```

#### np.mgrid[]  .ravel() np.c_[]

三个函数连用，构成网格坐标点

- `np.mgrid[]`
  `np.mgrid[起始值:结束值:步长,起始值:结束值:步长,...]`

- `x.ravel()` 将x变成一维数组，把.前变量拉直

- `np.c_[]` 使返回的间隔数值点配对
  `np.c_[数组1，数组2]`

  例如：

```python
import numpy as np
x,y = np.mgrid[1:3:1,2:4:0.5]
grid = np.c_[x.ravel(),y.ravel()]
print("x:",x)
print("y:",y)
print('grid:\n',grid)
```

#### 损失函数api

```python
tf.losses.categorical_crossentropy(y_,y)
loss_ce1 = tf.losses.categorical_crossentropy([1,0],[0.6,0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1,0],[0.8,0.2])
print("loss_ce1:",loss_ce1)
print("loss_ce2:",loss_ce2)
```

softmax与交叉熵结合
输出先过softmax函数，再计算y与y

```python
tf.nn.softmax_corss_entropy_with_logits(y_,y)
y_ = np.array(1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0)   #y_表示label

y = np.array(12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6.1)
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_corssentropy(y_,y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_,y)
```
