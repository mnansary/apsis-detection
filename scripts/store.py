#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------

import os
import tensorflow as tf 
from tqdm import tqdm
from glob import glob 
import cv2 
import argparse
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# number of images to store in a tfrecord
DATA_NUM  = 128
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_tfrecord(image_paths,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        
        image_paths     :   specific number of image paths	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in image_paths:
            
            char_path=str(image_path).replace('images','heatmaps')
            word_path=str(image_path).replace('images','linkmaps')
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            # char
            with(open(char_path,'rb')) as fid:
                char_bytes=fid.read()
            
            # word
            with(open(word_path,'rb')) as fid:
                word_bytes=fid.read()
            
            
            
            data ={ 'image':_bytes_feature(image_bytes),
                    'heatmap':_bytes_feature(char_bytes),
                    'linkmap':_bytes_feature(word_bytes)
            }
            
            
            
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(_paths,mode_dir):
    '''	        
        tf record wrapper
        args:	        
            _paths    :   all image paths for a mode	        
            mode_dir  :   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(_paths),DATA_NUM)):
        # paths
        image_paths= _paths[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(image_paths,mode_dir,r_num)    



def main(args):
    save_path =create_dir(args.save_dir,"tfrecords")
    save_path =create_dir(save_path,args.ds_iden)
    _paths=[img_path for img_path in tqdm(glob(os.path.join(args.data_dir,"*.*")))]
    genTFRecords(_paths,save_path)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("TFRecords Data Creation Script")
    parser.add_argument("data_dir", help="Path to the images folder")
    parser.add_argument("save_dir", help="Path to save the tfrecords")
    parser.add_argument("ds_iden", help="dataset Identifier")
    args = parser.parse_args()
    main(args)
