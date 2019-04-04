import tensorflow as tf
import  numpy as np
"""
0.数据读取
1.构建词库
2.数据封装
3.构建LSTM网络
"""

class LSTM(object):
    def __init__(self):
        pass
    def LSTM(self,input):
        lstm=tf.nn.rnn_cell.BasicRNNCell(num_units=10)
        out,h=tf.nn.dynamic_rnn(lstm,input,)

