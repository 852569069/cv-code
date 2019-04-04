import tensorflow as tf
import os
from PIL import Image
import numpy as np
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
        self.labels_train=self.labels[self.indicator:end_indicator]
        self.indicator=end_indicator
        if self.indicator==len(self.labels):
            self.indicator=0
            p=np.random.permutation(len(self.labels))
            self.labels=np.array(self.labels)[p]
            self.img_data=np.array(self.img_data)[p]

        return self.img_data_train,self.labels_train


class conv2d(object):
    def __init__(self):
        self.batch_size = 50
        self.channel=[32,64,128]
        self.conv2d_layer()
        pass
    def conv2d_layer(self):
        input_place=tf.placeholder(tf.float32,[self.batch_size,3072])
        input_1=tf.reshape(input_place,[self.batch_size,3,32,32])
        input_img=tf.transpose(input_1,[0,2,3,1])
        input=input_img
        with tf.variable_scope('conv2d',reuse=tf.AUTO_REUSE):
            for i in range(2):
                conv2d_data=tf.layers.conv2d(input,self.channel[i],
                                             kernel_size=[3,3],
                                             strides=[1,1],
                                             activation=tf.nn.relu)

                conv2d_data=tf.layers.conv2d(conv2d_data,self.channel[i],
                                             kernel_size=[1,1],
                                             strides=[1,1],
                                             activation=tf.nn.relu)
                conv2d_data=tf.layers.batch_normalization(conv2d_data)
                conv2d_max_pool=tf.layers.max_pooling2d(conv2d_data,pool_size=[2,2],strides=[2,2])
                input=conv2d_max_pool
            flat=tf.layers.flatten(input)
            dense1=tf.layers.dense(flat,1024)
            # dense2=tf.layers.dense(dense1,256)
            dense1=tf.layers.dropout(dense1)
            self.dense3=tf.layers.dense(dense1,10,activation=tf.nn.softmax)
            # tf.summary.histogram('con',conv2d_data)
            self.vari=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='conv2d')
        return input_place,self.dense3,input_img

    def loss(self):
        label_place=tf.placeholder(tf.int32,[self.batch_size,])
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_place,
                                                                                logits=self.dense3))
        acc=tf.metrics.accuracy(labels=label_place,predictions=tf.argmax(self.dense3,axis=1))
        tf.summary.scalar('acc',acc[-1])
        return label_place,self.loss,acc

    def train(self):
        train_op=tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return train_op
path=os.path.join('C:\\Users\\85256\OneDrive\学习资料\软件学习\深度学习\code\cv\data\cifar 10 data','data_batch_1')
data_Deal=Data_dael(path)

con=conv2d()
input_plac,pre,images=con.conv2d_layer()
label_place,loss,acc=con.loss()
train_op=con.train()

with tf.Session() as sess:
    write=tf.summary.FileWriter('test2',sess.graph)
    merged=tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(100000):
        data, labels = data_Deal.batch_data()
        fetch=[loss,train_op,acc,merged,images]
        dat=sess.run(fetch,{input_plac:data,label_place:labels})
        write.add_summary(dat[-2],i+1)
        print(dat[0])
        print((dat[-3][-1])*100)
        # print(dat[-1][0])
        # plt.imshow(dat[-1][0])
        plt.show()



