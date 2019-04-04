
"""
1.数据输入
2.构建计算图
3.训练数据
"""
import tensorflow as tf

from PIL import Image
from tensorflow import gfile
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
learning_rate=0.002
beta1=0.5
class DCGAN(object):
    def __init__(self,batch_size,z_dim=100,img_size=32,):
        self.z_dim=z_dim
        self.img_size=img_size
        self.indicator=0
        self.example_num=55000
        self.z_data=np.random.standard_normal([55000,4])
        self.load_data()
        self.batch_size=batch_size
        self.resize_img()
        self.random_shuffle()
    def load_data(self):
        self.img_data=mnist.train.images
    def random_shuffle(self):
        p=np.random.permutation(55000)
        self.img_data=self.img_data[p]
        self.z_data=self.z_data[p]
    def resize_img(self):
        data=self.img_data*255
        img_data_array=[]
        for i in range(55000):#为了计算的快捷，还需要修改。
            img_data=np.reshape(data[i],[28,28])
            img_data=Image.fromarray(img_data)
            img_data=img_data.resize([32,32])
            img_data=np.reshape(np.asarray(img_data)/127.5-1,[32,32,1])
            img_data_array.append(img_data)
        self.img_data_array=np.asarray(img_data_array)
    def next_batch(self):#很有意思的一个data_shuffle
        end_indicater=self.indicator+self.batch_size
        if end_indicater>55000:
            self.random_shuffle()
            self.indicator=0
            end_indicater=self.indicator+self.batch_size
        assert end_indicater<55000
        self.batch_data=self.img_data_array[self.indicator:end_indicater]
        self.batch_z_data=self.z_data[self.indicator:end_indicater]
        self.indicator=end_indicater
        return np.asarray(self.batch_z_data,np.float32),\
               np.asarray(self.batch_data,np.float32)
#以上的部分用于获取数据
#一个卷积和一个反卷积。
def conv2d_tranpose(input,out_channel,name,training=True,with_bn_relu=False):
    with tf.name_scope(name):
        conv2d_trans=tf.layers.conv2d_transpose(input,out_channel,
                                                    [5,5],
                                                    strides=[2,2],padding='SAME')
        if with_bn_relu:
            bn=tf.layers.batch_normalization(conv2d_trans,training=training)
            return bn
        else:
            return conv2d_trans
def conv2d(input,out_channel,name,training=True):
    def leak_relu(x,rate=0.2,name=''):
        return tf.maximum(x,rate*x,name=name)
    with tf.variable_scope(name):
        conv2d_out=tf.layers.conv2d(input,out_channel,[5,5],[2,2],
                                padding='SAME')
        bn=tf.layers.batch_normalization(conv2d_out,training=training)#暂时没有理解training的意义。
        conv2d_data= leak_relu(bn,name='output')
        tf.summary.histogram('conv2d',conv2d_out)
        return conv2d_data

#一个生成器和一个判别器。

class generator(object):#这里就将类变成了一个可调用对象。和函数的性质差不多。
    def __init__(self,init_conv_size,training=True):
        self.init_conv_size=init_conv_size
        self.g_conv_channel=[128,64,32,1]
        self.d_conv_channel=[32,64,125,256]
        self.training=training
        self.reuse=False

    def __call__(self,inputs):
        """"
        生成器，一开始生成的是一个长度为4的向量。通过一个全连接层，转为初始卷积大小：【4，4，1】.
        #然后将初始的[4,4]的一堆卷积通过反卷积放大。其通道数的变化是g_conv_channel=[128,64,32,1]。
         但是同时每经历一次反卷积，其size加倍。所以4*4-->8*8-->16*16-->32*32。所以最后输出的size是【32，32，1】的结构的。
        卷积，每次处理完都会跟着一个批归一化。对于卷积是每一层后面都会跟着一个批归一化。
        """""
        input= tf.convert_to_tensor(inputs)
        with tf.variable_scope('generators',reuse=self.reuse):
            with tf.variable_scope('input_conv-1',reuse=self.reuse):
                fc=tf.layers.dense(input,self.g_conv_channel[0]*
                                   self.init_conv_size*self.init_conv_size)
                conv0=tf.reshape(fc,[-1,self.init_conv_size,
                                     self.init_conv_size,
                                     self.g_conv_channel[0]])
                bn0=tf.layers.batch_normalization(conv0)
                relu0=tf.nn.relu(bn0)
                deconv=relu0
                for i in range(1,len(self.g_conv_channel)):
                    with_bn_relu=(i !=len(self.g_conv_channel)-1)
                    deconv=conv2d_tranpose(deconv,self.g_conv_channel[i],
                                           'deconv%d'%i,self.training,with_bn_relu)
                img_input=deconv
                with tf.variable_scope('img'):
                    img=tf.nn.tanh(img_input,'img')
            self.reuse=True
            self.variable=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='generators')
        return img_input
class discrim(object):
    def __init__(self):
        self.d_conv_channel=[32,64,125,256]
        self.reuse=False
    def __call__(self,input):
        img_tensor=tf.convert_to_tensor(input)
        with tf.variable_scope('discrim',reuse=self.reuse):#32*32的图片-->16*16-->8*8-->4*4-->2*2。展开之后是一行4列。
            for i in range(len(self.d_conv_channel)):
                img_tensor=conv2d(img_tensor,self.d_conv_channel[i],name='conv2d_data%d'%i)
            img_tensor=img_tensor
        with tf.variable_scope('fc',reuse=self.reuse):
            flat=tf.layers.flatten(img_tensor)
            logit=tf.layers.dense(flat,2,name='logit')
        self.reuse=True
        self.variable=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='discrim')
        return logit

"""#将生成的数据和原始的数据开始做误差分析"""

class DCGAN_TM(object):
    def __init__(self):
        self.batch_size=128
        self.z_size=4
        self.img_size=32
        self.init_conv2_size=4
        self.generator=generator(self.init_conv2_size)
        self.discrim=discrim()
        pass
    def build(self):
        #build函数是用于建立一个误差的。输入是生成的【128，4】的一个向量，以及从数据集中取出来的【128，32，32，1】的图像。
        #输出的是三组loss.
        self.z_placeholder=tf.placeholder(tf.float32,[self.batch_size,self.z_size])
        self.img_placeholder=tf.placeholder(tf.float32,[self.batch_size,
                                                        self.img_size,
                                                        self.img_size,1])
        generator_img=self.generator(self.z_placeholder)#就是这个占位符中的数据，拿去生成的数据。
        #最后会生成一个[batch_size,32,32,1]的数据。然后把生成的假图像放到判别器中进行判别。
        fake_img_logit=self.discrim(generator_img)
        #生成出来的图片进行判别之后会生成【batch_size,2的结构】
        real_img_logit=self.discrim(self.img_placeholder)
        # #损失函数的考虑：
        #             #对生成器来说，生成的图片因该能让判别器判别为真。
        #             #对判别器来说，对生成的图片应该判别为假。
        #             #对于原来的图片应该判别为真。
        # #判别器的第一个loss,对真的图像应该判别为真。
        loss_on_real_to_real=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
                                                                    tf.ones(self.batch_size,
                                                                    dtype=tf.int64),logits=
                                                                    real_img_logit))
        # #判别器的第二个loss，对假的图片判别为假。
        loss_on_fake_to_fake=tf.cast(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
                                                                    tf.zeros(self.batch_size,
                                                                             dtype=tf.int64),logits=
                                                                            fake_img_logit)),tf.float32)
        #生成器的loss，让判别器判断出生成的图片为真。
        loss_on_fake_to_real=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
                                                                        tf.ones(self.batch_size,
                                                                        tf.int64),logits=
                                                                        fake_img_logit))
        #然后把这些loss进行打包，按照训练的步骤进行打包。
        tf.add_to_collection('g_loss',loss_on_fake_to_real)
        tf.add_to_collection('d_loss',loss_on_real_to_real)
        tf.add_to_collection('d_loss', loss_on_fake_to_fake)
        loss={'g':tf.add_n(tf.get_collection('g_loss'),name='g_total_loss'),
              'd':tf.add_n(tf.get_collection('d_loss'),name='d_total_loss')}
        tf.summary.scalar('loss',loss['g'])

        return loss,generator_img,self.z_placeholder,self.img_placeholder

    def build_op(self,losses,learning_rate,beta1):
        g_opt=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        d_opt =tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        g_opt_op=g_opt.minimize(losses['g'],var_list=self.generator.variable)
        d_opt_op=d_opt.minimize(losses['d'],var_list=self.discrim.variable)
        with tf.control_dependencies([g_opt_op,d_opt_op]):
            return tf.no_op(name='train')

def combie_show_imgs(img,img_size,rows=8,col=16):
    """"#把一个batch的小图片生成一张大图片。一共有8*16，128张图也就是batch_size"""""
    big_img=[]
    for i in range(rows):
        img_row=[]
        for l in range(col):
            data=img[i*8+l]
            data=np.reshape(data,[img_size,img_size])
            data=(data+1)*127.5
            img_row.append(data)
        img_row=np.hstack(img_row)
        big_img=big_img.append(img_row)
    big_img=np.vstack(big_img)
    big_img=np.asarray(big_img,np.uint8)#np.uint8表示2^8.取0-255之间的数值。
    big_img=Image.fromarray(big_img)
    return big_img


# 获取到了数据。z_data的数据用于生成图像，img——data的数据用于判断。
DC=DCGAN(128)

ge=generator(4)
sess=tf.Session()
# ge(z_data)
sess.run(tf.global_variables_initializer())
# fake_img=sess.run(ge(z_data))
# print(fake_img)
dcgan=DCGAN_TM()
losses,img_gen,z_placeholder,img_placeholder=dcgan.build()
tf.summary.image('gen_img',img_gen)
sess.run(tf.global_variables_initializer())
train_op=dcgan.build_op(losses,learning_rate=learning_rate,beta1=beta1)
write_data=tf.summary.FileWriter('log_office',sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(10000):
    z_data, img_data = DC.next_batch()
    merged=tf.summary.merge_all()
    fetches=[train_op,losses['g'],losses['d']]
    should_sample=(i+1)%10==0
    if should_sample:
        fetches+=[img_gen,merged]
    all=sess.run(fetches,{z_placeholder:z_data,img_placeholder:img_data})
    if should_sample:
        write_data.add_summary(all[-1],i+1)
        print(all[1])
    # if should_sample:
    #     plt.imshow((np.reshape(out_value[3][0],[32,32])+1)*127.5)
    #     plt.show()








