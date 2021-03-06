#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import sys
sys.path.extend([os.path.dirname(os.path.abspath(__file__))])
import cv2
import time
import numpy as np
import tensorflow as tf
import utils
from OneEuroFilter import OneEuroFilter



class VNectEstimator:
    # the side length of the bounding box
    _box_size = 368
    # this factor indicates that the input box size is 8 times the side length of the output heatmaps
    _hm_factor = 8
    # number of the joints to be detected
    _joints_num = 21
    # parent joint indexes of each joint (for plotting the skeletal lines)
    _joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

    def __init__(self):
        print('Initializing VNectEstimator...')
        # the ratio factors to scale the input image crops, no more than 1.0
        self.scales = [1] # to be consistent with the author when training
        # initialize one euro filters for all the joints
        config_2d = {
            'freq': 120,
            'mincutoff': 1.7,
            'beta': 0.3,
            'dcutoff': 1.0
        }
        config_3d = {
            'freq': 120,
            'mincutoff': 0.8,
            'beta': 0.4,
            'dcutoff': 1.0
        }
        self.filter_2d = [(OneEuroFilter(**config_2d), OneEuroFilter(**config_2d)) for _ in range(self._joints_num)]
        self.filter_3d = [(OneEuroFilter(**config_3d), OneEuroFilter(**config_3d), OneEuroFilter(**config_3d))
                          for _ in range(self._joints_num)]
                
        self.model = tf.keras.models.load_model('models/tf_model/vnect_tf')
        print('VNectEstimator initialized.')

    def __call__(self, img_input):
        t = time.time()
        img_batch = self._gen_input_batch(img_input, self._box_size, self.scales)
        # inference
        res = self.model.predict(img_batch)
        hm, x_hm, y_hm, z_hm = np.split(res, [21, 2*21, 3*21], axis=-1)
                
        # Average scale outputs
        hm_size = self._box_size // self._hm_factor
        hm_avg = np.zeros(shape=(hm_size, hm_size, self._joints_num))
        x_hm_avg = np.zeros(shape=(hm_size, hm_size, self._joints_num))
        y_hm_avg = np.zeros(shape=(hm_size, hm_size, self._joints_num))
        z_hm_avg = np.zeros(shape=(hm_size, hm_size, self._joints_num))
        
        
        for i in range(len(self.scales)):
            rescale = 1/self.scales[i]
            
            scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_x_hm = cv2.resize(x_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_y_hm = cv2.resize(y_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_z_hm = cv2.resize(z_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            
            
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            
            # TODO: This gives a dimentsion missmatch for me.
            #hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
            #          mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            #x_hm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
            #            mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            #y_hm_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
            #            mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            #z_hm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
            #            mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            
            # TODO: fix:
            rescale = hm_avg.shape[0] / scaled_hm.shape[0]
            hm_avg += cv2.resize(scaled_hm, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            x_hm_avg += cv2.resize(scaled_x_hm, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            y_hm_avg += cv2.resize(scaled_y_hm, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            z_hm_avg += cv2.resize(scaled_z_hm, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            
        hm_avg /= len(self.scales)
        x_hm_avg /= len(self.scales)
        y_hm_avg /= len(self.scales)
        z_hm_avg /= len(self.scales)
        
        joints_2d = utils.extract_2d_joints_from_heatmaps(hm_avg, self._box_size, self._hm_factor)
        joints_3d = utils.extract_3d_joints_from_heatmaps(joints_2d, x_hm_avg, y_hm_avg, z_hm_avg, self._hm_factor)
        joints_2d, joints_3d = self._joint_filter(joints_2d, joints_3d)
        #print('FPS: {:>2.2f}'.format(1 / (time.time() - t)))

        return joints_2d, joints_3d

    @staticmethod
    def _gen_input_batch(img_input, box_size, scales):
        # any input image --> sqrared input image acceptable for the model
        img_square = utils.img_scale_squarify(img_input, box_size)
        
        # generate multi-scale input batch
        input_batch = []
        for scale in scales:
            img = utils.img_reduce_padding(img_square, scale) if scale < 1 else img_square
            input_batch.append(img)
            
        # input image range: [0, 255) --> [-0.4, 0.6)
        input_batch = np.asarray(input_batch, dtype=np.float32) / 255 - 0.4
        return input_batch

    def _joint_filter(self, joints_2d, joints_3d):
        for i in range(self._joints_num):
            joints_2d[i, 0] = self.filter_2d[i][0](joints_2d[i, 0], time.time())
            joints_2d[i, 1] = self.filter_2d[i][1](joints_2d[i, 1], time.time())

            joints_3d[i, 0] = self.filter_3d[i][0](joints_3d[i, 0], time.time())
            joints_3d[i, 1] = self.filter_3d[i][1](joints_3d[i, 1], time.time())
            joints_3d[i, 2] = self.filter_3d[i][2](joints_3d[i, 2], time.time())
        return joints_2d, joints_3d
    



if __name__ == '__main__':
    estimator = VNectEstimator()
    j_2d, j_3d = estimator(cv2.imread('../pic/test_pic.jpg'))
    print('\njoints_2d')
    for i, j in enumerate(j_2d):
        print(i, j)
    print('\njoints_3d')
    for i, j in enumerate(j_3d):
        print(i, j)
