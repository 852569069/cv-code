import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



a=np.random.randint(598,605,[100,100])
b=np.random.randint(759,765,[100,100])
c=np.random.randint(1170,1175,[100,100])
d=np.zeros([100,100])

#100,400.的数据。
data1=np.hstack([a,d,b,c,np.zeros([100,1],dtype=np.int32)])#1
data2=np.hstack([a,b,d,c,np.zeros([100,1],dtype=np.int32)+1])
data3=np.hstack([a,b,c,d,np.zeros([100,1],dtype=np.int32)+2])
data_all=np.vstack([data1,data2,data3])

class data(object):
    """
    construct data_batch and shuffle.
    :param input:data need to seprate.
    :param batch:creat batch_data.
    :param data_spe: only used when creat batch

    """
    def __init__(self,batch_szie=10):
        self.indicator=0
        self.batch_size=batch_szie

    def __data_spe(self,input):
        x_data = []
        y_data = []
        for i in input:
            x_data.append(list(i[0:400]))
            y_data.append(i[-1])
        return list(x_data), y_data
    def batch(self,input):
        if self.indicator==len(input):
            np.random.shuffle(input)
            x_data, y_data = self.__data_spe(input)
            self.indicator=0
        x_data,y_data=self.__data_spe(input)
        end_indicator=self.indicator+self.batch_size
        x_batch=x_data[self.indicator:end_indicator]
        y_batch=y_data[self.indicator:end_indicator]
        self.indicator=end_indicator
        y_batch=np.reshape(y_batch,[self.batch_size,1])
        return x_batch,y_batch


def con1d():
    input_pla=tf.placeholder(tf.float32,[10,400])
    labels=tf.placeholder(tf.int32,[10,1])
    input=tf.reshape(input_pla,[10,400,1])
    con1d=tf.layers.conv1d(input,32,1)
    con1d=tf.layers.max_pooling1d(con1d,2,2)
    con1d=tf.layers.conv1d(con1d,64,1)
    con1d=tf.layers.max_pooling1d(con1d,2,2)
    con1d=tf.layers.flatten(con1d)
    dese=tf.layers.dense(con1d,500,activation=tf.nn.relu)
    # dese=tf.layers.dropout(dese)
    dese=tf.layers.dense(dese,100,activation=tf.nn.sigmoid)
    dese=tf.layers.dense(dese,3,activation=tf.nn.softmax)
    loss=tf.losses.sparse_softmax_cross_entropy(labels,dese)
    acc=tf.metrics.accuracy(labels,tf.argmax(dese,axis=1))
    train_op=tf.train.AdamOptimizer(0.0001).minimize(loss)
    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    for i in range(10000):
        d = data()
        x, y = d.batch(data_all)
        result=sess.run([train_op,acc,loss],{input_pla:x,labels:y})
        if i%300==0:
            saver.save(sess,'./ckp/ckp%d'%i)
        print(result[-1])
        print(result[-2][-1])

if __name__ == '__main__':
    con = con1d()
