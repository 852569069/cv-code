import tensorflow as tf
import numpy as np
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""dcgan网络"""
#模块1，生成数据，一个是图片，另外一个是2*2的随机向量。
"""
0 没有池化层，卷积层的步数为2
1 在生成器和判别器中都使用bn
2 深层架构中移除全连接层。
3 生成器中，全都使用relu，最后一层使用tanh。
4 判别器中全部使用leak——relu。
"""

g_channel=[256,128,64,3]
d_channel=[32,64,128,256]


class data_gen(object):
    def __init__(self):
        self.log = 'D:\BaiduNetdiskDownload\img_align_celeba'
        self.dir = os.listdir(self.log)
        self.batch_size=128
        self.z_size=4
        self.indicator=0
        self.img_gen()
        # self.next_batch()
        self.shuffle()
    def img_gen(self):#生成数据
        self.all_data = []
        for i in self.dir[0:1000]:
            path = os.path.join(self.log, i)
            img = Image.open(path)
            print('already loaded %s' % i)
            data = np.array(img,np.float32)/127.5-1
            self.all_data.append(data)
        self.z_data=np.random.normal(0,1,[len(self.all_data),4]).astype(np.float32)


    def shuffle(self):
        p=np.int32(np.random.permutation(len(self.all_data)))
        self.all_data=np.array(self.all_data)[p]

    def next_batch(self):
        if self.indicator>10000:
            self.shuffle()
            self.indicator=0
        self.end_indicator=self.indicator+self.batch_size
        batch_data=self.all_data[0:self.end_indicator]
        batch_z_data=self.z_data[0:self.end_indicator]
        self.indicator=self.end_indicator
        batch_data=np.pad(batch_data,[[0,0],[19,19],[39,39],[0,0]],'constant')
        return batch_data,batch_z_data
class discrim(object):
    def __init__(self):
        self.batch_size = 128
        self.if_reuse=False
        self.d_channel = [32, 64, 128, 256]
    def __call__(self,input):
        with tf.variable_scope('dis',reuse=tf.AUTO_REUSE):
            input=tf.convert_to_tensor(input)
            tf.summary.image('images',input,max_outputs=5)
            for i in range(4):
                conv2d_d=tf.layers.conv2d(input,filters=self.d_channel[i],
                                          kernel_size=[3,3],strides=[2,2],
                                          activation=tf.nn.leaky_relu,
                                          padding='SAME')
                input=tf.layers.batch_normalization(conv2d_d)
            data=input
            data=tf.layers.flatten(data)
            data=tf.layers.dense(data,2)
            self.varibel=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='dis')
            return data

class gen(object):
    def __init__(self):
        self.batch_size = 128
        self.g_channel=[1024,512,256,128,3]
        self.if_relu=tf.nn.relu
        self.init_deconv2d_len=16
        self.init_deconv2d_width=16
        pass
    def __call__(self,input):
        with tf.variable_scope('gen',reuse=tf.AUTO_REUSE):
            input=tf.convert_to_tensor(input)
            deconv2d_data=tf.layers.dense(input,
                                          self.init_deconv2d_len*
                                           self.init_deconv2d_width*(self.g_channel[0]))

            deconv2d_data=tf.reshape(deconv2d_data,[self.batch_size,self.init_deconv2d_len,
                                        self.init_deconv2d_width,self.g_channel[0]])
            self.g_channel.remove(1024)
            for i in range(len(self.g_channel)):
                if i+1==len(self.g_channel):
                    self.if_relu=tf.nn.tanh
                deconv2d=tf.layers.conv2d_transpose(deconv2d_data,g_channel[i],
                                                    [5,5],strides=[2,2],
                                                    padding='SAME',
                                                    activation=self.if_relu)
                deconv2d_bn=tf.layers.batch_normalization(deconv2d)
                deconv2d_data=deconv2d_bn

            deconv2d_final=deconv2d_bn
            tf.summary.image('img_gen', deconv2d_final)
            self.vari=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gen')
            return deconv2d_final


class dcgan(object):
    #此时需要将生成的图片送给判别器，还需要把原始的图片给判别器。
    def __init__(self):
        self.g=gen()
        self.d=discrim()
        pass
    def train_net(self,img_batch_data, z_data):
        g = gen()
        d = discrim()
        fake_img_gen = g(z_data)
        real_img_dis = d(img_batch_data)
        fake_img_dis = d(fake_img_gen)
        #判别器中，真图判别为真
        loss_on_real_to_real=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones(128,tf.int32),logits=real_img_dis))
        #生成器：让假的图判别为真。
        loss_on_fake_to_real=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones(128,tf.int32),logits=fake_img_dis))
        # #判别器：让假的图片判别为假图。
        loss_on_fake_to_fake=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros(128,tf.int32),logits=fake_img_dis))
        tf.add_to_collection('g_loss',loss_on_fake_to_real)
        tf.add_to_collection('d_loss',loss_on_fake_to_fake)
        tf.add_to_collection('d_loss',loss_on_real_to_real)
        loss={'g':tf.add_n(tf.get_collection('g_loss'),name='g_total_loss'),
                'd':tf.add_n(tf.get_collection('d_loss'),name='d_total_loss')}

        return loss

    def train(self):
        return gen().vari

        # g_train_op=tf.train.AdamOptimizer(0.0001).minimize(loss['g'],var_list=self.g.vari)
        # d_train_op=tf.train.AdamOptimizer(0.0001).minimize(loss['d'],var_list=self.d.varibel)
        # with tf.control_dependencies(g_train_op,d_train_op):
        #     return tf.no_op(name='train')

        # return g_total_loss,d_total_loss
        # with tf.Session() as sess:
        #     for i in range(1):
        #         img_batch_data,z_data=data_g.next_batch()
        #         sess.run(tf.global_variables_initializer())
        #         fake_img=img_generator(z_data)
        #         # real_img_dis=discr(img_batch_data)
        #         # fake_img_dis=discr(fake_img)


dc=dcgan()
data_g = data_gen()
img_batch_data, z_data = data_g.next_batch()
loss=dc.train_net(img_batch_data, z_data)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1):
    img_batch_data, z_data = data_g.next_batch()
    train_all=dc.train()
    fetch=[loss,train_all]
    result=sess.run(fetch)
    print(result[0])






# print(sess.run([loss,loss1]))

