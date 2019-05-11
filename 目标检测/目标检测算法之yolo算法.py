import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import data_load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
1.从xml读取出框，输出为7*7*25的格式。

"""











class Yolo(object):
    def __init__(self):
        self.s = 7
        self.b = 2
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]

        # 对于图片上的每一个点生成对应的坐标。
        self.x_offset = np.transpose(np.reshape([np.arange(
            self.s)] * self.b * self.s, [self.b, self.s, self.s]), [1, 2, 0])  # 转置部分不清楚原理。纵向排序。
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])  # 横向排列
        self.thrshold = 0.2
        self.c = len(self.classes)
        self.box_num = 2
        self.max_out = 10
        self.img_size = [448, 448]
        self.iou_threshold=0.5



    def build_net(self):
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])
        with tf.variable_scope('yolo'):
            with tf.variable_scope('conv_2'):
                net = tf.layers.conv2d(self.images, 64, 7, 2)
            net = tf.layers.max_pooling2d(net, 2, 2)
            with tf.variable_scope('conv_4'):
                net = tf.layers.conv2d(net, 192, 3, 1)
            net = tf.layers.max_pooling2d(net, 2, 2)
            with tf.variable_scope('conv_6'):
                net = tf.layers.conv2d(net, 128, 1, 1)
            with tf.variable_scope('conv_7'):
                net = tf.layers.conv2d(net, 256, 3, 1)
            with tf.variable_scope('conv_8'):
                net = tf.layers.conv2d(net, 256, 1, 1)
            with tf.variable_scope('conv_9'):
                net = tf.layers.conv2d(net, 512, 3, 1)
            net = tf.layers.max_pooling2d(net, 2, 2)
            with tf.variable_scope('conv_11'):
                net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
            with tf.variable_scope('conv_12'):
                net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
            with tf.variable_scope('conv_13'):
                net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
            with tf.variable_scope('conv_14'):
                net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
            with tf.variable_scope('conv_15'):
                net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
            with tf.variable_scope('conv_16'):
                net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
            with tf.variable_scope('conv_17'):
                net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
            with tf.variable_scope('conv_18'):
                net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
            with tf.variable_scope('conv_19'):
                net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
            with tf.variable_scope('conv_20'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
                net = tf.layers.max_pooling2d(net, 2, 2)
            with tf.variable_scope('conv_22'):
                net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
            with tf.variable_scope('conv_23'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
            with tf.variable_scope('conv_24'):
                net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
            with tf.variable_scope('conv_25'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
            with tf.variable_scope('conv_26'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
            with tf.variable_scope('conv_28'):
                net = tf.layers.conv2d(net, 1024, 3, 2, padding='same')
            with tf.variable_scope('conv_29'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
            with tf.variable_scope('conv_30'):
                net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')

            net = tf.layers.flatten(net)
            with tf.variable_scope('fc_33'):
                net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            with tf.variable_scope('fc_34'):
                net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            with tf.variable_scope('fc_36'):
                net = tf.layers.dense(net, self.s * self.s * (self.c + self.b * 5))
        return net, self.images
    # 前面的卷积存在一些问题，此处输出的结果是【？，7*7*（20+2+(4+1）*2】
    # 仅仅预测框内只有一种物体的情况。所以7*7*（20+2（4+1）），也就是说，两个anchor box的物体分类考虑为一样的。

    def build_detector(self,predict):
        """将pre转为可以直接画图的数据"""
        conf=tf.reshape(predict[:,0:7*7*2*1],[-1,7,7,2,1])
        box=tf.reshape(predict[:,7*7*2*1:7*7*2*1+7*7*2*4],[-1,7,7,2,4])
        box=tf.stack((box[:,:,:,:,0]+self.x_offset)/7*448,
                   box[:,:,:,:,1]+self.y_offset/7*448,
                   box[:,:,:,:,2]*448,
                   box[:,:,:,:,3]*448)

        classes=tf.reshape(predict[:,(7*7*2*1+7*7*2*2+7*7*2*4):],[-1,7,7,1,20])

        score=conf*classes
        box=tf.reshape(box,[-1,4])
        score=tf.reshape(score,[-1,4])
        classes_id=tf.argmax(score,axis=1)
        score=tf.reduce_mean(score,axis=1)

        mask=score>self.thrshold

        score=tf.boolean_mask(score,mask)
        box=tf.boolean_mask(box,mask)
        classes_id=tf.boolean_mask(classes_id,mask)

        x=box[:,0]
        y=box[:,1]
        w=box[:,2]
        h=box[:,3]
        boxes=tf.stack(x-w//2,y-h//2,x+w//2,y+h//2)

        num_index=tf.image.non_max_suppression(boxes,score,
                                               max_output_size=self.max_out)
        return num_index




        # self.width=tf.placeholder(tf.float32,name='img_w')
        # self.height=tf.placeholder(tf.float32,name='img_h')
        # index1=self.s*self.s*self.c
        # index2=index1+self.s*self.s*self.b
        #
        # class_prob=tf.reshape(predict[:,0:index1],[self.s,self.s,self.c])#尺度为[?,7,7,20]
        #
        # conf=tf.reshape(predict[:,index1:index2],[self.s,self.s,self.b])#[?,7,7,2]
        #
        # box=tf.reshape(predict[:,index2:],[self.s,self.s,self.b,4])#[?,7,7,2,4]
        # #知道坐标之后，转换为相对于图片的左上角的相对坐标。其转换的公式有待理解。
        # #拉了一泡尿，突然就他妈的想明白了，相对坐标的转化。一开始计算出来的是每一个方格内部的相对坐标，加上方格排列好的【0，1，2，3，。。。】，这样就可以得到以方格为坐标的位置。
        # #然后，我们除以方格的长度，再乘以图片的长宽，得到的就是相对原图的位置，也就是说，以像素点为坐标的位置。
        # x_re=(box[:,:,:,0]+tf.constant(self.x_offset,dtype=tf.float32))/self.s*self.width
        # y_re=(box[:,:,:,1]+tf.constant(self.y_offset,dtype=tf.float32))/self.s*self.height
        # w_re=tf.square(box[:,:,:,2])*self.width
        # y_re=tf.square(box[:,:,:,3])*self.height
        # box_new=tf.stack([x_re,y_re,w_re,y_re],axis=3)
        # #此处有毛病，【7，7，2，4】与x_offset【7，7，2】,输出的维度是pre的输出的维度是含有batch_size的，但是此处的运算里不能存在。
        # score=tf.expand_dims(conf,-1)*tf.expand_dims(class_prob,2)#此处相乘是conf的值与类别进行相乘，这样，
        # # 得到一个【7，7，2，20】的值。也就是说，每个box对应20个类别的概率。
        # score=tf.reshape(score,[-1,self.c])#经过reshape之后，得到[s*s*b,c]的结果。
        # box=tf.reshape(box,[-1,4])
        #
        # box_class=tf.argmax(score,axis=1)#找到两个box对应的20个类别中最大的一个概率的那个下标。[s*s*b,1]
        # box_class_score=tf.reduce_mean(score,axis=1)#平均得分。得到[s*s*b,1]的结果。
        #
        # #通过阈值过滤。tf.boolean_mask是通过布尔值来实现的，
        # # 如果对应位置的布尔值为true，就保留。
        # filter=box_class_score>=self.thrshold#
        # score=tf.boolean_mask(box_class_score,filter)#平均得分保留超过阈值的那部分。
        # box=tf.boolean_mask(box,filter)#box也只保留平均得分超过阈值的部分
        # box_class=tf.boolean_mask(box_class,filter)#box_class，也只保留平均得分超过阈值的部分。
        #
        #
        # _boxes = tf.stack([box[:, 0] - 0.5 * box[:, 2], box[:, 1] - 0.5 * box[:, 3],
        #                    box[:, 0] + 0.5 * box[:, 2], box[:, 1] + 0.5 * box[:, 3]], axis=1)
        #
        #
        # #得到真实的box框的左上角，右下角位置的坐标。
        #
        # nms_indices = tf.image.non_max_suppression(_boxes, score,
        #                                            self.max_out, self.thrshold)
        # #返回的是索引。
        #
        # self.scores = tf.gather(score, nms_indices)
        # self.boxes = tf.gather(box, nms_indices)
        # self.box_classes = tf.gather(box_class, nms_indices)
        # return self.width,self.height,self.boxes

    # def layer_loss(self,predict,labels):

    # def build_dec(self, predict):
    #     """输入：predict
    #         输出：box，class（类别），score（相当于原始的truth中的置信度）
    #     """
    #     index0 = self.s * self.s * self.c
    #     index1 = self.s * self.s * self.c + self.s * self.s * self.b
    #
    #     classes = tf.reshape(predict[:, 0:index0], [ 7, 7, 20])
    #     conf = tf.reshape(predict[:, index0:index1], [ 7, 7, 2])
    #     boxes = tf.reshape(predict[:, index1:], [ 7, 7, 2, 4])
    #
    #     boxes = tf.stack([(boxes[:, :, :, 0] +
    #                        self.x_offset) /
    #                       7 *
    #                       self.img_size[0], (boxes[ :, :, :, 1] +
    #                                          self.y_offset) /
    #                       7 *
    #                       self.img_size[0], boxes[ :, :, :, 2] *
    #                       self.img_size[0], boxes[ :, :, :, 3] *
    #                       self.img_size[0]])
    #     #这里先默认这样操作不会发生什么问题。
    #     boxes = tf.reshape(boxes, [ self.s * self.s * self.b, 4])
    #
    #     score = tf.expand_dims(classes, -2) * tf.expand_dims(conf, -1)
    #
    #     score = tf.reshape(score, [ self.s * self.s * self.b, self.c])
    #
    #     box_class = tf.argmax(score, axis=1)
    #
    #     averg_score = tf.reduce_mean(score, axis=1)
    #
    #     filter_mask = averg_score > self.thrshold
    #
    #     box_class = tf.boolean_mask(box_class, filter_mask)
    #     boxes=tf.boolean_mask(boxes,filter_mask)
    #     averg_score = tf.boolean_mask(averg_score, filter_mask)
    #
    #     _boxes=tf.stack([boxes[:,0]-boxes[:,2]//2,
    #                      boxes[:,1]-boxes[:,3]//2,
    #                      boxes[:,0]+boxes[:,2]//2,
    #                      boxes[:,1]+boxes[:,3]//2],axis=1)
    #     num_indexs=tf.image.non_max_suppression(_boxes,averg_score,max_output_size=self.max_out,iou_threshold=self.iou_threshold)
    #
    #     boxes=tf.gather(boxes,num_indexs)
    #     box_class=tf.gather(box_class,num_indexs)
    #     score=tf.gather(score,num_indexs)
    #
    #     return boxes,box_class,score

    def from_img(self, img):  # 载入图片数据。
        img_data = cv2.imread(img)
        # img_rgb=cv2.cvtColor(img_data,cv2.COLOR_RGB2BGR)
        img_resize = cv2.resize(img_data, (448, 448))  # 不包含通道。
        img_array = np.array(img_resize)
        img_array = np.expand_dims(img_array, 0)
        return img_array

    def draw_rec(self, img, box, color, score=0.88, classes=None, thick=2):
        for i in range(len(box)):
            x = box[i][0].astype(int)
            y = box[i][1].astype(int)
            w = box[i][2].astype(int) // 2
            h = box[i][3].astype(int) // 2
            img = np.squeeze(img)
            img = cv2.resize(img, (960, 540))
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), [0, 255, 0], 2)
            cv2.putText(img, str(score), (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def loss(self,predict,labels,scope='loss_layer'):
        # with tf.variable_scope(scope):

    def train_net(self):
        load=data_load.Load_data()
        pos,img_data=load.xml_load()
        pre, img_plac = self.build_net()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        fetch=[pre]
        pre_data=sess.run(fetch,{img_plac:img_data[0]})
        predict=tf.convert_to_tensor(pre_data)

        num_index=self.build_detector(predict)

        return pre_data





y = Yolo()
z=y.train_net()
print(z)
# # test_img = y.from_img('test.jpg')
# pre, img_plac = y.build_net()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# saver=tf.train.Saver()
# saver.restore(sess,'YOLO_small.ckpt')
# pre_1 = sess.run(pre, {img_plac: test_img})
#
# print(np.array(pre_1).shape)
# #
# #
# m = tf.convert_to_tensor(pre_1)
# # #
# result = y.build_dec(m)
# z = sess.run(result)
# print(z)
# print(np.array(z.shape))
#
