#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Run this code to build and save tensorflow model with corresponding weight values for VNect
"""


import os
import tensorflow as tf
from src.caffe2pkl import caffe2pkl
from src.vnect_model import VNect

#tf.compat.v1.disable_eager_execution()


def init_tf_weights(pfile, spath, model):
    # configurations
    PARAMSFILE = pfile
    SAVERPATH = spath

    if not tf.io.gfile.exists(SAVERPATH):
        tf.io.gfile.makedirs(SAVERPATH)

    #saver = tf.compat.v1.train.Saver()
    #with tf.compat.v1.Session() as sess:
        
    model.load_weights(PARAMSFILE)
    model.getModel().save(os.path.join(SAVERPATH, 'vnect_tf')) 
        #saver.save(sess, os.path.join(SAVERPATH, 'vnect_tf'))


# caffe model basepath
caffe_bpath = './models/caffe_model'
# caffe model files
prototxt_name = 'vnect_net.prototxt'
caffemodel_name = 'vnect_model.caffemodel'
# pickle file name
pkl_name = 'params.pkl'
pkl_file = os.path.join(caffe_bpath, pkl_name)
# tensorflow model path
tf_save_path = './models/tf_model'

if not os.path.exists(pkl_file):
    caffe2pkl(caffe_bpath, prototxt_name, caffemodel_name, pkl_name)

model = VNect()
init_tf_weights(pkl_file, tf_save_path, model)
