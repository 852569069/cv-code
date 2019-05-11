#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday August 14 15:50:00 2018
@author: CW
"""
import numpy as np
import matplotlib.pyplot as plt

boxes = np.array([[100, 100, 210, 210, 0.71],
                  [250, 250, 420, 420, 0.8],
                  [220, 200, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])
'''
boxes = np.array([[3, 6, 9, 11, 0.9],
                  [6, 3, 8, 7, 0.6],
                  [3, 7, 10, 12, 0.7],
                  [1, 4, 13, 7, 0.2]])
'''


def iou(xmin, ymin, xmax, ymax, areas, lastInd, beforeInd, threshold):
    # 将lastInd指向的box，与之前的所有存活的box指向坐标做比较
    xminTmp = np.maximum(xmin[lastInd], xmin[beforeInd])
    yminTmp = np.maximum(ymin[lastInd], ymin[beforeInd])
    xmaxTmp = np.minimum(xmax[lastInd], xmax[beforeInd])
    ymaxTmp = np.minimum(ymax[lastInd], ymax[beforeInd])

    # 计算lastInd指向的box，与其他box交集的，所有width，height
    width = np.maximum(0.0, xmaxTmp - xminTmp + 1)
    height = np.maximum(0.0, ymaxTmp - yminTmp + 1)

    # 计算存活box与last指向box的交集面积
    intersection = width * height
    union = areas[beforeInd] + areas[lastInd] - intersection
    iou_value = intersection / union

    indexOutput = [item[0] for item in zip(beforeInd, iou_value) if item[1] <= threshold]

    return indexOutput


def nms(boxes, threshold):
    assert isinstance(boxes, np.ndarray)
    assert boxes.shape[1] == 5

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    scores = boxes[:, 4]

    # calc area of each box
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    # score each box in ascending order
    scoresSorted = sorted(list(enumerate(scores)), key=lambda item: item[1])
    # save index
    index = [item[0] for item in scoresSorted]

    keep = []
    while len(index) > 0:
        lastInd = index[-1]
        keep.append(lastInd)

        # calc the iou of the last box and all the boxes before it
        index = iou(xmin, ymin, xmax, ymax, areas, lastInd, index[:-1], threshold)
    return keep


def bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title("after nms")


if __name__ == '__main__':
    # before nms
    bbox(boxes, 'k')
    remain = nms(boxes, threshold=0.6)
    # after nms
    bbox(boxes[remain], 'r')