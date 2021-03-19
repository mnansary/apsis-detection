# -*-coding: utf-8 -
'''
    @author: (Re-Moduling) MD. Nazmuddoha Ansary 
'''
#--------------------
# imports
#--------------------

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
#--------------------
# blocks
#--------------------
class UpsampleLike(keras.layers.Layer):
    """ 
        Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            return tf.compat.v1.image.resize_bilinear(source,size=(target_shape[1], target_shape[2]),half_pixel_centers=True)

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            return (input_shape[0][0], ) + input_shape[1][1:3] + (input_shape[0][-1], )

#--------------------
# up-conv implemented from paper
#--------------------

def upconv(x, n, filters):
    x = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=f'upconv{n}.conv.0')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.1')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.2')(x)
    x = keras.layers.Conv2D(filters=filters // 2,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            name=f'upconv{n}.conv.3')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.4')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.5')(x)
    return x








def build_efficientnet_backbone(inputs,weights):
    '''
        uses efficientnet b7 
    '''
    backbone = tf.keras.applications.EfficientNetB7(include_top=False,
                                                    input_tensor=inputs,
                                                    weights=weights)
    return [
        backbone.get_layer(slice_name).output for slice_name in [
            'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation',
            'block5a_expand_activation'
        ]
    ]


def model(input_shape=(512,512,3),weights=None):
    '''
        creates the craft model
        args:
            input_shape   :   the input shape of each image (defalut:(512,512,3))
            weights       :   weights for transfer learning (default:None, Available:imagenet)
        returns:
            a tf.keras model 
    '''
    inputs = keras.layers.Input(input_shape)

    s1, s2, s3, s4 = build_efficientnet_backbone(inputs=inputs,weights=weights)

    s5 = keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same',
                                   name='basenet.slice5.0')(s4)
    s5 = keras.layers.Conv2D(1024,
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             dilation_rate=6,
                             name='basenet.slice5.1')(s5)
    s5 = keras.layers.Conv2D(1024,
                             kernel_size=1,
                             strides=1,
                             padding='same',
                             name='basenet.slice5.2')(s5)

    y = keras.layers.Concatenate()([s5, s4])
    y = upconv(y, n=1, filters=512)
    y = UpsampleLike()([y, s3])
    y = keras.layers.Concatenate()([y, s3])
    y = upconv(y, n=2, filters=256)
    y = UpsampleLike()([y, s2])
    y = keras.layers.Concatenate()([y, s2])
    y = upconv(y, n=3, filters=128)
    y = UpsampleLike()([y, s1])
    y = keras.layers.Concatenate()([y, s1])
    features = upconv(y, n=4, filters=64)

    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                            name='conv_cls.0')(features)
    y = keras.layers.Activation('relu', name='conv_cls.1')(y)
    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                            name='conv_cls.2')(y)
    y = keras.layers.Activation('relu', name='conv_cls.3')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                            name='conv_cls.4')(y)
    y = keras.layers.Activation('relu', name='conv_cls.5')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same',
                            name='conv_cls.6')(y)
    y = keras.layers.Activation('relu', name='conv_cls.7')(y)
    y = keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same',
                            name='conv_cls.8')(y)
    y = keras.layers.Activation('sigmoid')(y)
    model = keras.models.Model(inputs=inputs, outputs=y)
    
    return model




