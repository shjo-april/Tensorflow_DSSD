import numpy as np
import tensorflow as tf

import resnet_v2.resnet_v2 as resnet_v2

from Define import *

initializer = tf.contrib.layers.xavier_initializer()

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'upconv2d')
        
        if bn:
            x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def Decode_Layer(offset_bboxes, anchors):
    with tf.variable_scope('DSSD_Decode'):
        # 1. offset_bboxes
        tx = offset_bboxes[..., 0]
        ty = offset_bboxes[..., 1]
        tw = offset_bboxes[..., 2]
        th = offset_bboxes[..., 3]
        
        # 2. anchors
        wa = anchors[..., 2] - anchors[..., 0]
        ha = anchors[..., 3] - anchors[..., 1]
        xa = anchors[..., 0] + wa / 2
        ya = anchors[..., 1] + ha / 2

        # 3. pred_bboxes (cxcywh)
        x = tx * wa + xa
        y = ty * ha + ya
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        
        # 4. pred_bboxes (cxcywh -> xyxy)
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2

        # 5. exception (0 ~ IMAGE_WIDTH , IMAGE_HEIGHT)
        xmin = tf.clip_by_value(xmin[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
        ymin = tf.clip_by_value(ymin[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
        xmax = tf.clip_by_value(xmax[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
        ymax = tf.clip_by_value(ymax[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
        
        pred_bboxes = tf.concat([xmin, ymin, xmax, ymax], axis = -1)

    return pred_bboxes

def build_head_loc(x, is_training, num_anchors, name, depth = 3):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_bn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        x = conv_bn_relu(x, num_anchors * 4, (3, 3), 1, 'same', is_training, 'loc', bn = False, activation = False)
    return x

def build_head_cls(x, is_training, num_anchors, name, depth = 3):
    with tf.variable_scope(name):
        for i in range(depth):
            x = conv_bn_relu(x, 256, (3, 3), 1, 'same', is_training, '{}'.format(i))
        x = conv_bn_relu(x, num_anchors * CLASSES, (3, 3), 1, 'same', is_training, 'cls', bn = False, activation = False)
    return x

def DSSD_ResNet_50(input_var, is_training, reuse = False):
    dssd_dic = {}
    dssd_sizes = []

    x = input_var - [103.939, 123.68, 116.779]
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, is_training = is_training, reuse = reuse)

    # for key in end_points.keys():
    #     print(key, end_points[key])
    # input()

    pyramid_dic = {}
    feature_maps = [end_points['resnet_v2_50/block{}'.format(i)] for i in [4, 2, 1]]
    
    with tf.variable_scope('DSSD', reuse = reuse):
        x = feature_maps[0]
        
        x = conv_bn_relu(x, 256, (1, 1), 1, 'valid', is_training, 'conv1')
        pyramid_dic['P3'] = x

        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'conv2')
        pyramid_dic['P4'] = x
        
        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'conv3')
        pyramid_dic['P5'] = x

        x = conv_bn_relu(pyramid_dic['P3'], 256, (3, 3), 2, 'same', is_training, 'upconv1', upscaling = True)
        x = conv_bn_relu(feature_maps[1], 256, (1, 1), 1, 'valid', is_training, 'sum_conv1') + x
        pyramid_dic['P2'] = x

        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'upconv2', upscaling = True)
        x = conv_bn_relu(feature_maps[2], 256, (1, 1), 1, 'valid', is_training, 'sum_conv2') + x
        pyramid_dic['P1'] = x
        
        '''
        # P1 : Tensor("DSSD/add_1:0", shape=(?, 64, 64, 256), dtype=float32)
        # P2 : Tensor("DSSD/add:0", shape=(?, 32, 32, 256), dtype=float32)
        # P3 : Tensor("DSSD/conv1/relu:0", shape=(?, 16, 16, 256), dtype=float32)
        # P4 : Tensor("DSSD/conv2/relu:0", shape=(?, 8, 8, 256), dtype=float32)
        # P5 : Tensor("DSSD/conv3/relu:0", shape=(?, 4, 4, 256), dtype=float32)
        '''
        for i in range(1, 5 + 1):
            print('# P{} :'.format(i), pyramid_dic['P{}'.format(i)])
        # input()
        
        pred_bboxes = []
        pred_classes = []
        
        anchors_per_location = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        
        for i in range(1, 5 + 1):
            feature_map = pyramid_dic['P{}'.format(i)]
            _, h, w, c = feature_map.shape.as_list()
            dssd_sizes.append([w, h])
            
            _pred_bboxes = build_head_loc(feature_map, is_training, anchors_per_location, 'P{}_bboxes'.format(i))
            _pred_classes = build_head_cls(feature_map, is_training, anchors_per_location, 'P{}_classes'.format(i))

            _pred_bboxes = tf.reshape(_pred_bboxes, [-1, h * w * anchors_per_location, 4])
            _pred_classes = tf.reshape(_pred_classes, [-1, h * w * anchors_per_location, CLASSES])
            
            pred_bboxes.append(_pred_bboxes)
            pred_classes.append(_pred_classes)

        pred_bboxes = tf.concat(pred_bboxes, axis = 1, name = 'bboxes')
        pred_classes = tf.concat(pred_classes, axis = 1, name = 'classes')

        dssd_dic['pred_bboxes'] = pred_bboxes
        dssd_dic['pred_classes'] = tf.nn.softmax(pred_classes, axis = -1)

    return dssd_dic, dssd_sizes

DSSD = DSSD_ResNet_50

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [8, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    
    ssd_dic, ssd_sizes = DSSD(input_var, False)
    
    print(ssd_dic['pred_bboxes'])
    print(ssd_dic['pred_classes'])
