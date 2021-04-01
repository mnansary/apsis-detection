# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''

# -*- coding: utf-8 -*-
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import os
import random
import tensorflow as tf
import cv2
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from .utils import LOG_INFO,create_dir
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#---------------------------------------------------------------
class Processor(object):
    def __init__(self,
                data_path,
                save_path,
                data_size=1024):
        '''
            initializes the class
            args:
                data_path   =   location of raw data folder which contains test and train folder
                save_path   =   location to save outputs 
                data_size   =   the size of tfrecords
                
        '''
        # public attributes
        self.data_path  =   data_path
        self.save_path  =   save_path
        self.data_size  =   data_size
        # private attributes
        self.__train_path   =   os.path.join(self.data_path,'train')
        self.__test_path    =   os.path.join(self.data_path,'test')
        
        # output paths
        self.__tfrec_path   =   create_dir(self.save_path,'tfrecords')
        self.__tfrec_train  =   create_dir(self.__tfrec_path,'train')
        self.__tfrec_test   =   create_dir(self.__tfrec_path,'test')
        # image paths
        self.__train_img_paths  =   [img_path for img_path in tqdm(glob(os.path.join(self.__train_path,"img","*.*")))]
        self.__test_img_paths   =   [img_path for img_path in tqdm(glob(os.path.join(self.__test_path,"img","*.*")))]
        
        
    def __toTfrecord(self):
        '''
        Creates tfrecords from Provided Image Paths
        '''
        tfrecord_name=f'{self.__rnum}.tfrecord'
        tfrecord_path=os.path.join(self.__rec_path,tfrecord_name) 
        LOG_INFO(tfrecord_path)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:    
            
            for img_path in tqdm(self.__paths):
                # img
                with(open(img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # textmap
                tmap_path=img_path.replace("img","textmap")
                with(open(tmap_path,'rb')) as fid:
                    tmap_png_bytes=fid.read()
                
                # linkmap
                lmap_path=img_path.replace("img","linkmap")
                with(open(lmap_path,'rb')) as fid:
                    lmap_png_bytes=fid.read()
                
                # feature desc
                data ={ 'image'  :_bytes_feature(image_png_bytes),
                        'textmap':_bytes_feature(tmap_png_bytes),
                        'linkmap':_bytes_feature(lmap_png_bytes)                
                }
                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            
    def __create_df(self):
        '''
            tf record wrapper
        '''
        for idx in range(0,len(self.__img_paths),self.data_size):
            self.__paths      =   self.__img_paths[idx:idx+self.data_size]
            self.__rnum       =   idx//self.data_size
            self.__toTfrecord()

    def process(self):
        '''
            routine to create output
        '''
        # create tf recs
        ## train
        self.__img_paths=self.__train_img_paths
        self.__rec_path =self.__tfrec_train
        self.__create_df()
        ## test 
        self.__img_paths=self.__test_img_paths
        self.__rec_path =self.__tfrec_test
        self.__create_df()
        
        
