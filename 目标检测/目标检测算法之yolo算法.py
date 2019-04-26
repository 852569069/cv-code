import tensorflow as tf
import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    def build_net(self):
        self.images = tf.placeholder(tf.float32, [50, 448, 448, 3])
        net = tf.layers.conv2d(self.images, 64, 7, 2)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 192, 3, 1)
        net = tf.layers.max_pooling2d(net, 2, 2)
        # 输出【】

        net = tf.layers.conv2d(net, 128, 1, 1)
        net = tf.layers.conv2d(net, 256, 3, 1)
        net = tf.layers.conv2d(net, 256, 1, 1)
        net = tf.layers.conv2d(net, 512, 3, 1)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 256, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 512, 1, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.conv2d(net, 1024, 3, 1, padding='same')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 512, activation=tf.nn.relu)
        net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
        net = tf.layers.dense(net, self.s * self.s * (self.c + self.b * 5))
        return net, self.images
    # 前面的卷积存在一些问题，此处输出的结果是【？，7*7*（20+2+2*4）】

    def build_detector(self, predict):
        self.width = tf.placeholder(tf.float32, name='img_w')
        self.height = tf.placeholder(tf.float32, name='img_h')
        index1 = self.s * self.s * self.c
        index2 = index1 + self.s * self.s * self.b

        class_prob = tf.reshape(predict[:, 0:index1], [
                                self.s, self.s, self.c])  # 尺度为[?,7,7,20]

        conf = tf.reshape(predict[:, index1:index2], [
                          self.s, self.s, self.b])  # [?,7,7,2]

        box = tf.reshape(predict[:, index2:], [
                         self.s, self.s, self.b, 4])  # [?,7,7,2,4]
        # 知道坐标之后，转换为相对于图片的左上角的相对坐标。其转换的公式有待理解。
        # 拉了一泡尿，突然就他妈的想明白了，相对坐标的转化。一开始计算出来的是每一个方格内部的相对坐标，加上方格排列好的【0，1，2，3，。。。】，这样就可以得到以方格为坐标的位置。
        # 然后，我们除以方格的长度，再乘以图片的长宽，得到的就是相对原图的位置，也就是说，以像素点为坐标的位置。
        x_re = (box[:, :, :, 0] + tf.constant(self.x_offset,
                                              dtype=tf.float32)) / self.s * self.width
        y_re = (box[:, :, :, 1] + tf.constant(self.y_offset,
                                              dtype=tf.float32)) / self.s * self.height
        w_re = tf.square(box[:, :, :, 2]) * self.width
        y_re = tf.square(box[:, :, :, 3]) * self.height
        box_new = tf.stack([x_re, y_re, w_re, y_re], axis=3)
        # 此处有毛病，【7，7，2，4】与x_offset【7，7，2】,输出的维度是pre的输出的维度是含有batch_size的，但是此处的运算里不能存在。
        # 此处相乘是conf的值与类别进行相乘，这样，
        score = tf.expand_dims(conf, -1) * tf.expand_dims(class_prob, 2)
        # 得到一个【7，7，2，20】的值。也就是说，每个box对应20个类别的概率。
        score = tf.reshape(score, [-1, self.c])  # 经过reshape之后，得到[s*s*b,c]的结果。
        box = tf.reshape(box, [-1, 4])

        box_class = tf.argmax(score, axis=1)  # 找到两个box对应的20个类别中最大的一个概率的那个下标。
        box_class_score = tf.reduce_mean(score, axis=1)  # 平均得分。得到[s*s*b,1]的结果。

        # 通过阈值过滤。tf.boolean_mask是通过布尔值来实现的，
        # 如果对应位置的布尔值为true，就保留。
        filter = box_class_score >= self.thrshold
        score = tf.boolean_mask(box_class_score, filter)  # 平均得分保留超过阈值的那部分。
        box = tf.boolean_mask(box, filter)  # box也只保留平均得分超过阈值的部分
        box_class = tf.boolean_mask(
            box_class, filter)  # box_class，也只保留平均得分超过阈值的部分。

        _boxes = tf.stack([box[:, 0] -
                           0.5 *
                           box[:, 2], box[:, 1] -
                           0.5 *
                           box[:, 3], box[:, 0] +
                           0.5 *
                           box[:, 2], box[:, 1] +
                           0.5 *
                           box[:, 3]], axis=1)

        # 得到真实的box框的左上角，右下角位置的坐标。

        nms_indices = tf.image.non_max_suppression(_boxes, score,
                                                   self.max_out, self.thrshold)

        self.scores = tf.gather(score, nms_indices)
        self.boxes = tf.gather(box, nms_indices)
        self.box_classes = tf.gather(box_class, nms_indices)

        return self.width, self.height, self.boxes


test_img = np.random.random([50, 448, 448, 3])
y = Yolo()
pre, img_plac = y.build_net()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
pre_1 = sess.run(pre, {img_plac: test_img})


for i in pre_1:
    m = tf.reshape(tf.convert_to_tensor(i), [-1, 1470])
    w_plac, h_plac, box = y.build_detector(m)
    o = sess.run(box, {w_plac: 300, h_plac: 400})
    print(o)
