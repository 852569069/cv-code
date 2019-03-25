import tensorflow as tf
import  numpy as np

#构建啊LSTM. 0 构建cell 1 h0 和 input输入。
#调用：cell.call(input，h0)=output，h1.

# cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# h0=cell.zero_state(32,np.float32)
# input=tf.placeholder(np.float32,[32,100])
# output,h1=cell.call(input,h0)
#此处的h1表示的是一个状态，包含了H输出和C的输出
# print(cell.state_size)

#构建多层rnn。
def get_rnn():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

muti_rnn=tf.nn.rnn_cell.MultiRNNCell([get_rnn() for _ in range(3)])
# h0=muti_rnn.zero_state(32,np.float32)
input=tf.placeholder(np.float32,[10,32,100])
# muti_rnn_out,muti_rnn_h1=muti_rnn.call(input,h0)
# print
(muti_rnn.state_size)
output,state=tf.nn.dynamic_rnn(muti_rnn,input,
                               initial_state=muti_rnn.zero_state(32,np.float32),
                               time_major=True)
#需要初始化第一个state的状况。定好其batch——size就可以。state的维度就是【batch——size，num——units】
#输出的是最后一个的out。
print(output)


