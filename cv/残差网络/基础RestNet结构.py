import pickle
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Data_dael():
    def __init__(self, filename):
        self.batch_size = 50
        self.filename = filename
        self.load_file()
        self.indicator = 0
        pass

    def load_file(self):
        with open(self.filename, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        self.img_data = data[b'data']
        self.labels = data[b'labels']

    def batch_data(self):
        end_indicator = self.indicator + self.batch_size
        self.img_data_train = (
            self.img_data[self.indicator:end_indicator]) / 127.5 - 1
        self.img_data_train = np.reshape(
            self.img_data_train, [
                self.batch_size, 3, 32, 32])
        self.img_data_train = np.transpose(self.img_data_train, [0, 2, 3, 1])
        self.labels_train = self.labels[self.indicator:end_indicator]
        self.indicator = end_indicator
        if self.indicator == len(self.labels):
            self.indicator = 0
            p = np.random.permutation(len(self.labels))
            self.labels = np.array(self.labels)[p]
            self.img_data = np.array(self.img_data)[p]
        return self.img_data_train, self.labels_train


data_d = Data_dael('data_batch_1')
img, label = data_d.batch_data()


def reside():
    img_tensor = tf.convert_to_tensor(img)  # [50,32,32,3]
    # 一般的残差结构的dim和步长变化最大控制在2.

    conv1 = tf.layers.conv2d(
        img_tensor, 64, [
            3, 3], [
            2, 2], padding='same')  # [50,16,16,64]

    conv2 = tf.layers.conv2d(
        conv1, 128, [
            3, 3], [
            2, 2], padding='same')  # [50,8,8,128]
    conv3 = tf.layers.conv2d(
        conv2, 128, [
            3, 3], [
            1, 1], padding='same')  # [50,8,8,128]

    # 所以在conv1和conv2中间添加一个残差结构单元。此处的通道数加倍了，所以原来的通道要做pading。
    # 其次，feature_map的大小变为了原来的一半，所以原来的输入做一个pooling。
    x = tf.layers.max_pooling2d(conv1, strides=[2, 2], pool_size=[2, 2])
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [32, 32]])
    conv3 = conv3 + x
    return conv3


with tf.Session() as sess:
    p = reside()
    sess.run(tf.global_variables_initializer())
    l = sess.run(p)
    print(l)
    print(l.shape)
