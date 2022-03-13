# TensorFlow模型部署与保存  
***
- 1 keras\_save\_graph\_def\_and\_weights.ipynb  
- 2 keras\_load\_weights.ipynb  
- 3 keras\_saved\_model.ipynb  
- 4 signature\_to\_saved\_model.ipynb  
- 5 to\_concrete\_function.ipynb  
- 6 to\_tflite.ipynb  
- 7 tflite\_interpreter.ipynb  
- 8 to\_quantized\_tflite.ipynb
- 9 quantized\_tflite\_interpreter.ipynb  
- 10 tfjs\_converter.ipynb  
- 11 tfjs\_converter\_py.ipynb  
***
- keras-保存参数与保存模型+参数
- keras,签名函数到SavedModel
- keras,SavedModel,签名函数到具体函数
- keras,SvaedModel，具体函数到tflite
- Tflite量化
- tensorflow js部署模型
- Android部署模型
***
![tfdeploy6.png](https://github.com/XuLongjia/Tensorflow2.0Learning/blob/master/images/tfdeploy6.png)
***
![/tfdeploy1.png](https://github.com/XuLongjia/Tensorflow2.0Learning/blob/master/images/tfdeploy1.png)
***
## 模型保存
- 文件格式
    - checkpoint 与graphdef(tf1.0)
    - keras(hdf5),savedModel(tf2.0)
- 保存的是什么？
    - 参数
    - 网络结构
## TFLite
- TFLite Converter
    - 模型转化
- TFLite Interpreter
    - 模型加载
    - 支持android与IOS设备
    - 支持多种语言
- TFLite - Flatbuffer
    - Google开源的跨平台数据序列化库
    - 有点
        - 直接读取序列化数据
        - 高效的内存使用和速度，无需占用额外内存
        - 灵活，数据前后向兼容，灵活控制数据结构
        - 使用少量的代码即可完成功能
        - 强数据类型，易于使用
