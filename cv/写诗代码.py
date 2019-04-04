
import tensorflow as tf
import os
import collections
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        result_poem=[]
        for l in all_poem:
            change=l[1:-1]
            # l[0:-2]=change
            result_poem.append(change)
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
        embedding_vari=tf.random_normal([self.word_num,128])
        embedding_code=tf.nn.embedding_lookup(embedding_vari,input)

        basic_rnn=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
        """
        #一个很重要的点，这个num_unit可以理解为LSTM节点内的神经元的个数。
        # 可以参考：https://www.zhihu.com/question/64470274。所以输入的数据是【batch_size,80,128】的数据。
        # 其中的80是有80个time_step.
        # 然后128也就是我们embedding编码之后的维度，就必须和神经元的个数对应。
        """
        muti_lstm=tf.nn.rnn_cell.MultiRNNCell([basic_rnn for i in range(self.num_layer)])
        _,out=tf.nn.dynamic_rnn(muti_lstm,embedding_code,initial_state=muti_lstm.zero_state(self.batch_size,tf.float32))


        return input,out



data_deal=data_deal()
result,final_poem=data_deal.poem_len_control()
print(result[24])
print(final_poem[24])
# poem_num=len(result)
# auto_poem=auto_poem(poem_num)
# input_place,embedding=auto_poem.lstm(result)
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# all=sess.run(embedding,{input_place:result[0:100]})
# print(all)



# embedding_vari=tf.random_normal()
#
# tf.nn.embedding_lookup
