import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle

class Data_dael():
    def __init__(self,filename):
        self.batch_size=50
        self.filename=filename
        self.load_file()
        self.indicator = 0
        pass
    def load_file(self):
        with open(self.filename, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        self.img_data=data[b'data']
        self.labels=data[b'labels']
    def batch_data(self):
        end_indicator=self.indicator+self.batch_size
        self.img_data_train=(self.img_data[self.indicator:end_indicator])/127.5-1
        self.img_data_train=np.reshape(self.img_data_train,[self.batch_size,3,32,32])
        self.img_data_train=np.transpose(self.img_data_train,[0,2,3,1])
        self.labels_train=self.labels[self.indicator:end_indicator]
        self.indicator=end_indicator
        if self.indicator==len(self.labels):
            self.indicator=0
            p=np.random.permutation(len(self.labels))
            self.labels=np.array(self.labels)[p]
            self.img_data=np.array(self.img_data)[p]
        return self.img_data_train,self.labels_train

data_d=Data_dael('data_batch_1')
img, label = data_d.batch_data()


def reside(input,output_channel):
    #一般的残差结构的dim和步长变化最大控制在2.
    input_channel=input.get_shape().as_list()[-1]
    global stride,dim_incre
    if input_channel==output_channel:
        stride=[1,1]
        dim_incre=False
    elif input_channel*2==output_channel:
        stride=[2,2]
        dim_incre=True
    conv1=tf.layers.conv2d(input,output_channel,[3,3],
                           strides=stride,
                           padding='same',
                           activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, output_channel, [3, 3],
                             [1, 1], padding='same',
                             activation=tf.nn.relu)
    conv2=tf.layers.batch_normalization(conv2)
    conv2=tf.layers.dropout(conv2)

    #所以在input和conv2中间添加一个残差结构单元。此处的通道数加倍了，所以原来的通道要做pading。
    #其次，feature_map的大小变为了原来的一半，所以原来的输入做一个pooling。
    if dim_incre:
        input=tf.layers.max_pooling2d(input,strides=[2,2],pool_size=[2,2])
        x=tf.pad(input,[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]])
    else:
        x=input
    output=conv2+x
    return output


class Rest_net(object):
    def __init__(self):
        self.init_channel=16
        self.num_res=[2,2,]
    def model(self,input,labels):
        input=tf.placeholder(tf.float32,[50,32,32,3])
        labels=tf.placeholder(tf.int32,[50,])
        tf.summary.image('img',input,max_outputs=10)
        conv2d=tf.layers.conv2d(input,self.init_channel,[3,3],
                                strides=[1,1],padding='same',
                                activation=tf.nn.relu)
        #对于传入的数据先做一个简单的卷积操作。然后开始做残差块的操作。
        #残差块是可以进行两种操作的，一种是通道数不改变，另外一种是通道数加倍。

        for i in range(len(self.num_res)):
            for l in range(self.num_res[i]):
                conv2=reside(conv2d,
                            self.init_channel*(2**(i+1)))
        #认真考虑一下其中的结构，对于第一个残差（也就是【2，3，2】中的第一个2），
        # 里面有两个残差块，所以通道数是2^0,和2^1.
        #对于2^0，也就是通道数不发生改变，卷积步长也为1的卷积。
        #对于2^1,通道数加倍了，步长为2，feature _map的大小减半了。
        #所以变化过程应该是【50，32，32，3】--> ([50,32,32,16]-->[50,16,16,32]) -->([50,16,16,32]-->[50,8,8,64]-->[50,4,4,128])
                conv2d=conv2
        tf.summary.histogram('conv2d',conv2d)
        flat=tf.layers.flatten(conv2d)
        fc=tf.layers.dense(flat,10,activation=tf.nn.softmax)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=fc))
        acc=tf.metrics.accuracy(labels,tf.argmax(fc,axis=1))
        tf.summary.scalar('acc',acc[-1])
        return self.loss,acc[-1],input,labels

    def train(self):
        train_op=tf.train.AdamOptimizer(0.004).minimize(self.loss)
        return train_op



with tf.Session() as sess:
    r=Rest_net()
    loss, acc ,img_pla,label_plac= r.model(img,label)
    train_=r.train()
    write=tf.summary.FileWriter('test',sess.graph)
    meg = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(10000):
        img, label = data_d.batch_data()
        fetch=[loss,train_,meg,acc]
        l=sess.run(fetch,{img_pla:img,label_plac:label})
        megl=l[-2]
        write.add_summary(megl,i+1)
        print(l[0])
        print(l[-1])
