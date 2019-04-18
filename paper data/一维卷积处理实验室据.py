import tensorflow as tf
import os
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filename1='data11.csv'
filename2='data22.csv'
filename3='data33.csv'
filename4='data44.csv'
data1=pd.read_csv(filename4,header=None)

data1_=np.hstack([data1,np.zeros([279,1],dtype=np.int)])
data2=pd.read_csv(filename2,header=None)
data2_=np.hstack([data2,np.zeros([279,1],dtype=np.int)+1])
data3=pd.read_csv(filename3,header=None)
data3_=np.hstack([data3,np.zeros([279,1],dtype=np.int)+2])
data4=pd.read_csv(filename4,header=None)
data4_=np.hstack([data4,np.zeros([279,1],dtype=np.int)+3])
data_all=np.vstack([data1_,data2_,data3_,data4_])

class Data(object):
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
            x_data.append(list(i[0:50]))
            y_data.append(i[-1])
        return list(x_data), y_data
    def creat_text_data(self,input):
        np.random.shuffle(input)
        point=int(len(input)*0.8)
        train_data=input[0:point]
        test_data=input[-point:]
        return train_data,test_data

    def batch(self,input):
        if self.indicator>len(input)-10:
            np.random.shuffle(input)
            x_data, y_data = self.__data_spe(input)
            self.indicator=0
        x_data,y_data=self.__data_spe(input)
        end_indicator=self.indicator+self.batch_size
        x_batch=x_data[self.indicator:end_indicator]
        y_batch=y_data[self.indicator:end_indicator]
        self.indicator=end_indicator
        return x_batch,y_batch


def con1d():
    input_pla=tf.placeholder(tf.float32,[10,50])
    labels=tf.placeholder(tf.int32,[10,])
    input=tf.reshape(input_pla,[10,50,1])
    con1d=tf.layers.conv1d(input,32,1)
    con1d=tf.layers.batch_normalization(con1d)
    con1d=tf.layers.max_pooling1d(con1d,2,2)
    # con1d=tf.layers.conv1d(con1d,64,1)
    # con1d=tf.layers.max_pooling1d(con1d,2,2)
    con1d=tf.layers.flatten(con1d)
    dese=tf.layers.dense(con1d,500,activation=tf.nn.relu)
    # dese=tf.layers.dropout(dese)
    dese=tf.layers.dense(dese,100,activation=tf.nn.sigmoid)
    dese=tf.layers.dense(dese,4,activation=tf.nn.softmax)
    loss=tf.losses.sparse_softmax_cross_entropy(labels,dese)
    tf.summary.scalar('loss',loss)
    acc=tf.metrics.accuracy(labels,tf.argmax(dese,axis=1))
    tf.summary.scalar('acc',acc[-1])
    train_op=tf.train.AdamOptimizer(0.0006).minimize(loss)
    sess=tf.Session()
    write=tf.summary.FileWriter('test',sess.graph)
    meged=tf.summary.merge_all()
    saver=tf.train.Saver()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    d = Data()
    train_data, test_data = d.creat_text_data(data_all)
    for i in range(100000):
        x, y = d.batch(train_data)
        result=sess.run([train_op,acc,loss],{input_pla:x,labels:y})
        if i%300==0:
            all=sess.run(meged,{input_pla:x,labels:y})
            write.add_summary(all,i)
            saver.save(sess,'./ckp_TEST/ckp%d'%i)
        print(result[-1])
        print(result[-2][-1])

if __name__ == '__main__':
    con = con1d()
