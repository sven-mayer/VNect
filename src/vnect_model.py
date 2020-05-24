#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


# Reference:
# https://github.com/timctho/VNect-tensorflow


import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


class VNect:
    def __init__(self):
        self.is_training = False
        self.input_holder = tf.keras.layers.Input(shape=(368, 368, 3))
        self._build_network()

    def _build_network(self):
        # Conv
        self.conv1 = tf.keras.layers.Conv2D(64, (7,7), padding='same',  activation="relu", strides=(2,2), name='conv1')(self.input_holder)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same', name='pool1')(self.conv1)

        # Residual block 2a
        self.res2a_branch2a = tf.keras.layers.Conv2D(64, (1,1), padding='valid', activation="relu", name='res2a_branch2a')(self.pool1)
        self.res2a_branch2b = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu",name='res2a_branch2b')(self.res2a_branch2a)
        self.res2a_branch2c = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation=None, name='res2a_branch2c')(self.res2a_branch2b)
        self.res2a_branch1 = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation=None, name='res2a_branch1')(self.pool1)
        self.res2a = tf.math.add(self.res2a_branch2c, self.res2a_branch1, name='res2a_add')
        self.res2a = tf.nn.relu(self.res2a, name='res2a')

        # Residual block 2b
        self.res2b_branch2a = tf.keras.layers.Conv2D(64, (1,1), padding='valid', activation="relu", name='res2b_branch2a')(self.res2a)
        self.res2b_branch2b = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu", name='res2b_branch2b')(self.res2b_branch2a)
        self.res2b_branch2c = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation=None, name='res2b_branch2c')(self.res2b_branch2b)
        self.res2b = tf.math.add(self.res2b_branch2c, self.res2a, name='res2b_add')
        self.res2b = tf.nn.relu(self.res2b, name='res2b')

        # Residual block 2c
        self.res2c_branch2a = tf.keras.layers.Conv2D(64, (1,1), padding='valid', activation="relu", name='res2c_branch2a') (self.res2b)
        self.res2c_branch2b = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu", name='res2c_branch2b') (self.res2c_branch2a)
        self.res2c_branch2c = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation=None, name='res2c_branch2c')(self.res2c_branch2b)
        self.res2c = tf.math.add(self.res2c_branch2c, self.res2b, name='res2c_add')
        self.res2c = tf.nn.relu(self.res2c, name='res2c')

        # Residual block 3a
        self.res3a_branch2a = tf.keras.layers.Conv2D(128, (1,1), padding='valid', activation="relu", strides=(2,2), name='res3a_branch2a')(self.res2c)
        self.res3a_branch2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", name='res3a_branch2b') (self.res3a_branch2a)
        self.res3a_branch2c = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation=None, name='res3a_branch2c')(self.res3a_branch2b)
        self.res3a_branch1 = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation=None, strides=(2,2), name='res3a_branch1') (self.res2c)
        self.res3a = tf.math.add(self.res3a_branch2c, self.res3a_branch1, name='res3a_add')
        self.res3a = tf.nn.relu(self.res3a, name='res3a')

        # Residual block 3b
        self.res3b_branch2a = tf.keras.layers.Conv2D(128, (1,1), padding='valid', activation="relu", name='res3b_branch2a')(self.res3a)
        self.res3b_branch2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", name='res3b_branch2b')(self.res3b_branch2a)
        self.res3b_branch2c = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation=None, name='res3b_branch2c')(self.res3b_branch2b)
        self.res3b = tf.math.add(self.res3b_branch2c, self.res3a, name='res3b_add')
        self.res3b = tf.nn.relu(self.res3b, name='res3b')

        # Residual block 3c
        self.res3c_branch2a = tf.keras.layers.Conv2D(128, (1,1), padding='valid', activation="relu", name='res3c_branch2a')(self.res3b)
        self.res3c_branch2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", name='res3c_branch2b')(self.res3c_branch2a)
        self.res3c_branch2c = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation=None, name='res3c_branch2c')(self.res3c_branch2b)
        self.res3c = tf.math.add(self.res3c_branch2c, self.res3b, name='res3c_add')
        self.res3c = tf.nn.relu(self.res3c, name='res3c')

        # Residual block 3d
        self.res3d_branch2a = tf.keras.layers.Conv2D(128, (1,1), padding='valid', activation="relu", name='res3d_branch2a')(self.res3c)
        self.res3d_branch2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", name='res3d_branch2b')(self.res3d_branch2a)
        self.res3d_branch2c = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation=None, name='res3d_branch2c')(self.res3d_branch2b)
        self.res3d = tf.math.add(self.res3d_branch2c, self.res3c, name='res3d_add')
        self.res3d = tf.nn.relu(self.res3d, name='res3d')

        # Residual block 4a
        self.res4a_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", strides=(2,2), name='res4a_branch2a')(self.res3d)
        self.res4a_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4a_branch2b')(self.res4a_branch2a)
        self.res4a_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4a_branch2c')(self.res4a_branch2b)
        self.res4a_branch1 = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, strides=(2,2), name='res4a_branch1')(self.res3d)
        self.res4a = tf.math.add(self.res4a_branch2c, self.res4a_branch1, name='res4a_add')
        self.res4a = tf.nn.relu(self.res4a, name='res4a')

        # Residual block 4b
        self.res4b_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res4b_branch2a')(self.res4a)
        self.res4b_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4b_branch2b')(self.res4b_branch2a)
        self.res4b_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4b_branch2c')(self.res4b_branch2b)
        self.res4b = tf.math.add(self.res4b_branch2c, self.res4a, name='res4b_add')
        self.res4b = tf.nn.relu(self.res4b, name='res4b')

        # Residual block 4c
        self.res4c_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res4c_branch2a')(self.res4b)
        self.res4c_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4c_branch2b')(self.res4c_branch2a)
        self.res4c_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4c_branch2c')(self.res4c_branch2b)
        self.res4c = tf.math.add(self.res4c_branch2c, self.res4b, name='res4c_add')
        self.res4c = tf.nn.relu(self.res4c, name='res4c')

        # Residual block 4d
        self.res4d_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res4d_branch2a')(self.res4c)
        self.res4d_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4d_branch2b')(self.res4d_branch2a)
        self.res4d_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4d_branch2c')(self.res4d_branch2b)
        self.res4d = tf.math.add(self.res4d_branch2c, self.res4c, name='res4d_add')
        self.res4d = tf.nn.relu(self.res4d, name='res4d')

        # Residual block 4e
        self.res4e_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res4e_branch2a')(self.res4d)
        self.res4e_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4e_branch2b')(self.res4e_branch2a)
        self.res4e_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4e_branch2c')(self.res4e_branch2b)
        self.res4e = tf.math.add(self.res4e_branch2c, self.res4d, name='res4e_add')
        self.res4e = tf.nn.relu(self.res4e, name='res4e')

        # Residual block 4f
        self.res4f_branch2a = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res4f_branch2a')(self.res4e)
        self.res4f_branch2b = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation="relu", name='res4f_branch2b')(self.res4f_branch2a)
        self.res4f_branch2c = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res4f_branch2c')(self.res4f_branch2b)
        self.res4f = tf.math.add(self.res4f_branch2c, self.res4e, name='res4f_add')
        self.res4f = tf.nn.relu(self.res4f, name='res4f')
        #END ResNet50
            
        #Start model as described in paper Figure 5
        # Residual block 5a
        self.res5a_branch2a_new = tf.keras.layers.Conv2D(512, (1,1), padding='valid', activation="relu", name='res5a_branch2a_new')(self.res4f)
        self.res5a_branch2b_new = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation="relu", name='res5a_branch2b_new')(self.res5a_branch2a_new)
        self.res5a_branch2c_new = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res5a_branch2c_new')(self.res5a_branch2b_new)
        self.res5a_branch1_new = tf.keras.layers.Conv2D(1024, (1,1), padding='valid', activation=None, name='res5a_branch1_new')(self.res4f)
        self.res5a = tf.math.add(self.res5a_branch2c_new, self.res5a_branch1_new, name='res5a_add')
        self.res5a = tf.nn.relu(self.res5a, name='res5a')

        # Residual block 5b
        self.res5b_branch2a_new = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res5b_branch2a_new')(self.res5a)
        self.res5b_branch2b_new = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", name='res5b_branch2b_new')(self.res5b_branch2a_new)
        self.res5b_branch2c_new = tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation="relu", name='res5b_branch2c_new')(self.res5b_branch2b_new)

        # Transpose Conv
        self.res5c_branch1a = tf.keras.layers.Conv2DTranspose(3*21, (4,4), activation=None, strides=(2,2), padding='same', use_bias=False, name='res5c_branch1a')(self.res5b_branch2c_new)
        
        self.res5c_branch2a = tf.keras.layers.Conv2DTranspose(128, (4,4), activation=None,  strides=(2,2), padding='same', use_bias=False, name='res5c_branch2a')(self.res5b_branch2c_new)
        self.bn5c_branch2a = tf.keras.layers.BatchNormalization(scale=True, trainable=self.is_training, name='bn5c_branch2a')(self.res5c_branch2a)
        self.bn5c_branch2a = tf.nn.relu(self.bn5c_branch2a)
        
        # Start BL block in Figure 5 
        self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z = tf.split(self.res5c_branch1a, num_or_size_splits=3, axis=3)
        self.res5c_branch1a_sqr = tf.math.multiply(self.res5c_branch1a, self.res5c_branch1a, name='res5c_branch1a_sqr')
        self.res5c_delta_x_sqr, self.res5c_delta_y_sqr, self.res5c_delta_z_sqr = tf.split(self.res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
        self.res5c_bone_length_sqr = tf.math.add(tf.math.add(self.res5c_delta_x_sqr, self.res5c_delta_y_sqr), self.res5c_delta_z_sqr)
        self.res5c_bone_length = tf.math.sqrt(self.res5c_bone_length_sqr)
        # End BL block 
        
        # Gray White Black Black in Fig 5
        self.res5c_branch2a_feat = tf.concat([self.bn5c_branch2a, self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z, self.res5c_bone_length], axis=3, name='res5c_branch2a_feat')

        # Last two layers in Fig 5
        self.res5c_branch2b = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation="relu", bias_initializer='zeros', name='res5c_branch2b')(self.res5c_branch2a_feat)
        self.res5c_branch2c = tf.keras.layers.Conv2D(4*21, (1,1), padding='valid', activation=None, use_bias=False, name='res5c_branch2c')(self.res5c_branch2b)
        
        self.heatmap, self.x_heatmap, self.y_heatmap, self.z_heatmap = tf.split(self.res5c_branch2c, num_or_size_splits=4, axis=3)
        
        self.model = tf.keras.models.Model(self.input_holder, [self.heatmap, self.x_heatmap, self.y_heatmap, self.z_heatmap])
        self.model.summary()
        
        

    def load_weights(self, params_file):
        print("load_weights")
        # Read pretrained model file
        model_weights = pickle.load(open(params_file, 'rb'))

        df = pd.DataFrame(list(model_weights.keys()))
        df.columns = ["Key"]
        df["Layer"] = df.Key.apply(lambda x: x.split("/")[0])
        df["Value"] = df.Key.apply(lambda x: x.split("/")[1])
        
        for layername in df.Layer.unique():
            dfX = df[df.Layer == layername]
            lst = dfX.Value.unique()

            if len(lst) == 1:
                self.model.get_layer(layername).set_weights((model_weights['%s/kernel' % layername], ))
            elif len(lst) == 2:
                self.model.get_layer(layername).set_weights((model_weights['%s/kernel' % layername], model_weights['%s/bias' % layername]))
            elif len(lst) == 4:
                self.model.get_layer(layername).set_weights((model_weights['%s/gamma' % layername], model_weights['%s/beta' % layername], model_weights['%s/moving_mean' % layername], model_weights['%s/moving_variance' % layername]))
            else:
                print("WARNING: not used", layername, lst, self.model.get_layer(layername).get_weights())
    
    def getModel(self):
        return self.model

if __name__ == '__main__':
    model = VNect()
    print('VNect building successfully.')


