import tensorflow as tf
import numpy as np
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""dcgan网络"""
#模块1，生成数据，一个是图片，另外一个是2*2的随机向量。



g_channel=[256,128,64,3]
d_channel=[32,64,128,256]


class data_gen(object):
    def __init__(self):
        self.log = 'D:\BaiduNetdiskDownload\img_align_celeba'
        self.dir = os.listdir(self.log)
        self.batch_size=50
        self.indicator=0
        self.img_gen()
        # self.next_batch()
        self.shuffle()
    def img_gen(self):
        self.all_data = []
        for i in self.dir[0:1000]:
            path = os.path.join(self.log, i)
            img = Image.open(path)
            print('already loaded %s' % i)
            data = np.array(img,np.float32)/127.5-1
            self.all_data.append(data)
    def shuffle(self):
        p=np.int32(np.random.permutation(len(self.all_data)))
        self.all_data=np.array(self.all_data)[p]

    def next_batch(self):
        if self.indicator>10000:
            self.shuffle()
            self.indicator=0
        self.end_indicator=self.indicator+self.batch_size
        batch_data=self.all_data[0:self.end_indicator]
        self.indicator=self.end_indicator
        return batch_data





#
# def conv2d(input):
#     with tf.variable_scope('conv2d',reuse=tf.AUTO_REUSE):
#         for i in range(3):
#             conv2d_1=tf.layers.conv2d(input,32,kernel_size=[5,5],padding='same',strides=[2,2],activation=tf.nn.relu)
#             input=conv2d_1
#         return conv2d

# def deconv2d(input,name,channel):
#     with tf.variable_scope(name):
#         deconv2d=tf.layers.conv2d_transpose(input,channel,padding='same',kernel_size=[5,5],strides=[2,2],activation=tf.nn.relu)
#         return deconv2d





class discrim(object):
    def __init__(self):
        self.batch_size = 50
        self.if_reuse=False
        self.d_channel = [32, 64, 128, 256]
    def dis(self):
        conv2d_data=tf.placeholder(tf.float32,[50,218,178,3])
        with tf.variable_scope('dis',reuse=tf.AUTO_REUSE):
            input=conv2d_data
            tf.summary.image('images',input,max_outputs=5)
            for i in range(4):
                conv2d_d=tf.layers.conv2d(input,filters=self.d_channel[i],
                                          kernel_size=[3,3],strides=[2,2],
                                          padding='same')

                input=tf.layers.batch_normalization(conv2d_d)
            data=input



            return data,conv2d_data



discr=discrim()
out,input_place=discr.dis()
tf.summary.histogram('value',out)
data_g=data_gen()
img_batch_data=data_g.next_batch()

with tf.Session() as sess:
    writer=tf.summary.FileWriter('test',sess.graph)
    merged=tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    fetch=[out,merged]
    conv2d_out,merge=sess.run(fetch,{input_place:img_batch_data})
    writer.add_summary(merge)
    print(conv2d_out.shape)




