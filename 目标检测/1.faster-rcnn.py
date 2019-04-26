import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ssd(object):
    def __init__(self):
        pass
    def conv2d(self,x,filter,k_size,pading='same'):
        out=tf.layers.conv2d(x,filters=filter,
                             kernel_size=k_size,
                             padding=pading,
                             activation=tf.nn.relu)
    def set_net(self):
        img_pla=tf.placeholder(tf.float32,[None,300,300,3])
        with tf.variable_scope('ssd'):

