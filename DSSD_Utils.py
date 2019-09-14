# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from DataAugmentation import *

def get_data(xml_path, training, normalize = True, augment = True):
    if training:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = False)

        image = cv2.imread(image_path)
        
        if augment:
            image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

        image_h, image_w, image_c = image.shape
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

        gt_bboxes = gt_bboxes.astype(np.float32)
        gt_classes = np.asarray(gt_classes, dtype = np.int32)

        if normalize:
            gt_bboxes /= [image_w, image_h, image_w, image_h]
            gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

        # print(image.shape)
        # print(np.min(gt_bboxes), np.max(gt_bboxes), image.shape)

        # for bbox in gt_bboxes:
        #     print(bbox)
        #     xmin, ymin, xmax, ymax = (bbox * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]).astype(np.int32)
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # cv2.imshow('show', image)
        # cv2.waitKey(0)
    else:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = normalize)
        image = cv2.imread(image_path)

    return image, gt_bboxes, gt_classes

def generate_anchors(sizes, image_wh, anchor_scales, anchor_ratios):
    scales = np.linspace(0.1, 0.9, num = len(sizes))
    sizes = np.asarray(sizes, dtype = np.int32)
    image_wh = np.asarray(image_wh, dtype = np.float32)
    
    anchors = []
    for scale, size in zip(scales, sizes):
        # scales * ratios
        strides = image_wh / size
        # base_anchor_wh = strides # * 2
        base_anchor_wh = image_wh * scale
        base_anchor_whs = [base_anchor_wh * scale for scale in anchor_scales]
        
        '''
        [41 41] [15.65853659 15.65853659]
        [21 21] [30.57142857 30.57142857]
        [11 11] [58.36363636 58.36363636]
        [6 6] [107. 107.]
        [3 3] [214. 214.]
        [1 1] [642. 642.]
        '''
        # print(size, base_anchor_wh)
        
        anchor_wh_list = []
        for base_anchor_wh in base_anchor_whs:
            for anchor_ratio in anchor_ratios:
                w = base_anchor_wh[0] * np.sqrt(anchor_ratio)
                h = base_anchor_wh[1] / np.sqrt(anchor_ratio)
                anchor_wh_list.append([w, h])

        # append anchors 
        for y in range(size[1]):
            for x in range(size[0]):
                anchor_cx = (x + 0.5) * strides[0]
                anchor_cy = (y + 0.5) * strides[1]

                for anchor_wh in anchor_wh_list:
                    anchors.append([anchor_cx, anchor_cy] + anchor_wh)

    anchors = np.asarray(anchors, dtype = np.float32)

    cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]

    cond = np.logical_and(cx < IMAGE_WIDTH, cy < IMAGE_HEIGHT)
    cx, cy, w, h = cx[cond], cy[cond], w[cond], h[cond]

    xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    xmin = np.maximum(np.minimum(xmin, image_wh[0] - 1), 0.)
    ymin = np.maximum(np.minimum(ymin, image_wh[1] - 1), 0.)
    xmax = np.maximum(np.minimum(xmax, image_wh[0] - 1), 0.)
    ymax = np.maximum(np.minimum(ymax, image_wh[1] - 1), 0.)
    
    anchors = np.stack([xmin, ymin, xmax, ymax]).T
    return anchors

def Encode(gt_bboxes, gt_classes, anchors):
    encode_bboxes = np.zeros_like(anchors)
    encode_classes = np.zeros((anchors.shape[0], CLASSES), dtype = np.float32)
    encode_classes[:, 0] = 1.
    
    if len(gt_bboxes) != 0:
        # calculate ious
        ious = compute_bboxes_IoU(anchors, gt_bboxes)
        max_iou_indexs = np.argmax(ious, axis = 1)
        max_ious = ious[np.arange(anchors.shape[0]), max_iou_indexs]
        
        # get positive indexs
        positive_indexs = max_ious >= POSITIVE_IOU_THRESHOLD

        # set encode_classes
        positive_classes = gt_classes[max_iou_indexs][positive_indexs]
        encode_classes[positive_indexs, 0] = 0.
        encode_classes[positive_indexs, positive_classes] = 1.
        
        # set encode_bboxes
        encode_bboxes = gt_bboxes[max_iou_indexs]
    
    return encode_bboxes, encode_classes

if __name__ == '__main__':
    import cv2
    from DSSD import *

    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    dssd_dic, dssd_sizes = DSSD(input_var, False)

    anchors = generate_anchors(dssd_sizes, [IMAGE_WIDTH, IMAGE_HEIGHT], ANCHOR_SCALES, ANCHOR_RATIOS)

    # 1. Demo Anchors
    # bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)
    
    # for index, anchor in enumerate(anchors):
    #     xmin, ymin, xmax, ymax = anchor.astype(np.int32)
    
    #     cv2.circle(bg, ((xmax + xmin) // 2, (ymax + ymin) // 2), 1, (0, 0, 255), 2)
    #     cv2.rectangle(bg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    #     if (index + 1) % (len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)) == 0:
    #         cv2.imshow('show', bg)
    #         cv2.waitKey(1)

    #         bg = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)

    # 2. Demo GT bboxes (Encode -> Decode)
    xml_paths = glob.glob('D:/DB/VOC2007/train/xml/*.xml')
    
    for xml_path in xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = True)
        print(gt_bboxes, np.min(gt_bboxes), np.max(gt_bboxes), len(gt_bboxes))
        
        image = cv2.imread(image_path)
        h, w, c = image.shape
        
        encode_bboxes, encode_classes = Encode(gt_bboxes, gt_classes, anchors)
        positive_count = np.sum(encode_classes[:, 1:])
        
        positive_mask = np.max(encode_classes[:, 1:], axis = 1)
        positive_mask = positive_mask[:, np.newaxis]
        print(np.min(positive_mask), np.max(positive_mask))
        
        # (22890, 4) (22890, 21)
        print(np.min(positive_mask * encode_bboxes[:, :2]), np.max(positive_mask * encode_bboxes[:, :2]), \
              np.min(positive_mask * encode_bboxes[:, 2:]), np.max(positive_mask * encode_bboxes[:, 2:]))
        print(encode_bboxes.shape, encode_classes.shape, positive_count)

        pred_bboxes = []
        pred_classes = [] 

        positive_mask = np.max(encode_classes[:, 1:], axis = 1)
        for i, mask in enumerate(positive_mask):
            if mask == 1:
                xmin, ymin, xmax, ymax = (anchors[i] / [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT] * [w, h, w, h]).astype(np.int32)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

                pred_bbox = convert_bboxes(encode_bboxes[i], img_wh = [w, h])
                pred_class = np.argmax(encode_classes[i])
                pred_class = CLASS_NAMES[pred_class]

                pred_bboxes.append(pred_bbox)
                pred_classes.append(pred_class)

        for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = pred_bbox.astype(np.int32)

            cv2.putText(image, '{}'.format(pred_class), (xmin, ymin - 10), 1, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        cv2.imshow('show', image)
        cv2.waitKey(0)

