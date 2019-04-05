import tensorflow as tf
import os
import collections
import numpy as np
import tensorflow.contrib as con
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class data_deal(object):
    def __init__(self):
        self.dir='C:\\Users\85256\OneDrive\学习资料\软件学习\深度学习\code\cv\data\LSTM data'
        self.data_load()

    def data_load(self):
        all_file=os.listdir(self.dir)
        filename=os.path.join(self.dir,all_file[0])
        with open(filename,'r',encoding='utf-8') as f:
            data=f.readlines()
        all_poem=[]
        for poem in data:
            main_part=poem.strip().split(':')
            all_poem.append(main_part[-1])
        all_word=[]
        for m in all_poem:
            for n in m:
                all_word.append(n)
        counter=collections.Counter(all_word)
        word_id=dict(zip(counter,[i for i in range(len(counter))]))
        word_id['unk']=7559
        id_word=dict(zip([i for i in range(len(counter))],counter))
        self.all_poem_id=[]
        for i in all_poem:
            poem_code=[word_id.get(l)  for l in i]
            self.all_poem_id.append(poem_code)
        print(len(word_id))
        return self.all_poem_id

    """
    id转文字的操作。

    for l in all_poem_id:
        word=[id_word.get(m) for m in l]
        word=''.join(word)
        print(word)

    """

    def poem_len_control(self):
        all_poem=[]
        for i in self.all_poem_id[0:3759]:
            if len(i)>80:
                i=np.array(i[0:80])
                all_poem.append(i)
            elif len(i)>40 and len(i)<80:
                i=np.pad(np.array(i),[0,80-len(i)],'constant',constant_values=(7559))
                all_poem.append(i)

        result_poem=np.copy(all_poem)
        """
        此处注意,需要复制一份出来，
        否则会改变原来地址中的值，略抽象，慢慢体会。
        
        """
        for l in result_poem:
            change=l[1:]
            l[0:-1]=change
        return np.array(all_poem),result_poem


class auto_poem(object):
    def __init__(self,poem_num):
        self.batch_size=100
        self.embedding_size=128
        self.word_num=7600
        self.poem_num=poem_num
        self.num_layer=2
        pass
    def lstm(self,input):
        input=tf.placeholder(tf.int32,[self.batch_size,80])
        output=tf.placeholder(tf.int32,[self.batch_size,80])
        with tf.variable_scope('embedding',initializer=tf.random_uniform_initializer(-1,1)):
            embedding_vari=tf.get_variable('embedding',[self.word_num,128],tf.float32)
            embedding_code=tf.nn.embedding_lookup(embedding_vari,input)
        with tf.variable_scope('lstm',reuse=tf.AUTO_REUSE):
            basic_rnn=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
            """
            #一个很重要的点，这个num_unit可以理解为LSTM节点内的神经元的个数。
            # 可以参考：https://www.zhihu.com/question/64470274。所以输入的数据是【batch_size,80,128】的数据。
            # 其中的80是有80个time_step.
            # 然后128也就是我们embedding编码之后的维度，就必须和神经元的个数对应。
            """
            muti_lstm=tf.nn.rnn_cell.MultiRNNCell([basic_rnn for i in range(1)])
            out,_=tf.nn.dynamic_rnn(muti_lstm,embedding_code,
                                    initial_state=muti_lstm.zero_state
                                    (self.batch_size,tf.float32))

        #此处输出的结构时【100，80，128】，将其reshape成【-1，128】的结构的。也就是【8000，128】的结构。
        # 再连接一个全连接层。对应到分类上。

        out=tf.reshape(out,[-1,self.embedding_size])
        out=tf.layers.dense(out,self.word_num)
        #此时，输出的结构时【8000，7600】.前者的8000时batch_size*time_step.
        # 也就是说，将batch_size个诗句中的所有的词都合并在一起。而且对应的结构是，【【诗句1】【诗句2】【诗句3】。。。。】。
        # 其中诗句里面是【字1，字2，字3.。。。】。
        # 所以说里面的每一个元素都对应一个长度为【7600】的词库里面的一个词。
        # 可以参考：https://blog.csdn.net/xyz1584172808/article/details/83056179
        #output的结构是【100，80】，也就是说对应了正确的那个字。我们也将其做一个reshape。
        out_poem=tf.reshape(output,[-1])
        """
        此处必须传入list结构的tensor。
        List of 2D Tensors of shape [batch_size x num_decoder_symbols]
        """
        loss=con.legacy_seq2seq.sequence_loss_by_example(logits=[out],
                                                         targets=[out_poem],
                                                         weights=[tf.ones_like([8000,],
                                                        dtype=tf.float32)])
        self.loss=tf.reduce_mean(loss)

        self.tvars = tf.trainable_variables()
        self.grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,self.tvars),5)


        return input,output,self.loss,self.tvars
    def train_op(self):
        train=tf.train.AdamOptimizer(0.0001).apply_gradients(zip(self.grads,self.tvars))
        return train
data_deal=data_deal()
result,final_poem=data_deal.poem_len_control()

poem_num=len(result)
auto_poem=auto_poem(poem_num)
input_place,output_place,loss,tvar=auto_poem.lstm(result)
print(tvar)

for i in tvar:
    print(i)
train_op=auto_poem.train_op()

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    fetch=[loss,train_op]
    all=sess.run(fetch,{input_place:result[0:100],output_place:final_poem[0:100]})
    print(all[0])

