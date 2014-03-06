手写数字识别测试
================

#实现模型#

采用四层的卷积神经网络，自测的validate的精度为99.05%（80% training set, 20% validate set)

另外，尝试采用一层sparce autoencoder，精度为94.04%

#实验环境#

*   ubuntu 12.10 
*   python2.7
*   theano
*   numpy

实验中采用GPU加速：nvidia GTX650 

#源码及实现#

项目代码代管在github上，地址：https://github.com/Superjom/NeuralNetworks

模型实现位于: models/

具体训练及测试代码： apps/126

#预测命令#

./predict.sh "path-to-test-csv-file"

预测中，会*默认忽略csv文件的第一行*

#可能出现问题#

1. CUDO错误

    AttributeError: ("'NoneType' object has no attribute 'CudaNdarray'", <function CudaNdarray_unpickler at 0xbe5041c>,

这个是脱离原始nvidia的CUDA环境的问题，需要配置基于nvidia显卡的theano GPU运行环境:http://deeplearning.net/software/theano/tutorial/using_gpu.html。


