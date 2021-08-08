#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import numpy as np
import cv2
import random
#---------------------------------------------------------------
# common utils
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
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
# image utils
#---------------------------------------------------------------
def stripPads(arr,val):
  '''
      strip specific values
  '''
  arr=arr[~np.all(arr == val, axis=1)]
  arr=arr[:, ~np.all(arr == val, axis=0)]
  return arr
#---------------------------------------------------------------

def padLineImg(line_img,h_max,w_max):
    # shape
    h,w=line_img.shape
    # pad widths
    left_pad_width =(w_max-w)//2
    # print(left_pad_width)
    right_pad_width=w_max-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width))
    right_pad=np.zeros((h,right_pad_width))
    # pad
    line_img =np.concatenate([left_pad,line_img,right_pad],axis=1)
    
    # shape
    h,w=line_img.shape
    # pad heights
    top_pad_height =(h_max-h)//2
    bot_pad_height=h_max-h-top_pad_height
    # pads
    top_pad =np.zeros((top_pad_height,w))
    bot_pad=np.zeros((bot_pad_height,w))
    # pad
    line_img =np.concatenate([top_pad,line_img,bot_pad],axis=0)
    return line_img

def padAllAround(img,pad_dim):
    h,w=img.shape
    # pads
    left_pad =np.zeros((h,pad_dim))
    right_pad=np.zeros((h,pad_dim))
    # pad
    img =np.concatenate([left_pad,img,right_pad],axis=1)
    # shape
    h,w=img.shape
    top_pad =np.zeros((pad_dim,w))
    bot_pad=np.zeros((pad_dim,w))
    # pad
    img =np.concatenate([top_pad,img,bot_pad],axis=0)
    return img
#---------------------------------------------------------------
def placeWordOnMask(word,labeled_img,region_value,mask,ext_reg=False):
    '''
        @author
        places a specific image on a given background at a specific location
        args:
            word               :   greyscale image to place
            labeled_img        :   labeled image to place the image
            region_value       :   the specific value of the labled region
            mask               :   placement mask
        return:
            mak :   mask image after placing 'img'
    '''
    idx=np.where(labeled_img==region_value)
    h_li,w_li=labeled_img.shape
    # region
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    if ext_reg:
        h_reg = abs(y_max-y_min)
        w_reg = abs(x_max-x_min)
        # ext
        h_ext=int((random.randint(0,20)*h_reg)/100)
        w_ext=int((random.randint(0,20)*w_reg)/100)
        # region ext
        if y_min-h_ext>0:y_min-=h_ext # extend min height
        if y_max+h_ext<=h_li:y_max+=h_ext # extend max height
        if x_min-w_ext>0:x_min-=w_ext # extend min width
        if x_max+w_ext<=w_li:x_max+=w_ext # extend min width
    # resize image    
    h_max = abs(y_max-y_min)
    w_max = abs(x_max-x_min)
    word = cv2.resize(word, (w_max,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # place on mask
    mask[y_min:y_max,x_min:x_max]=word
    return mask
#---------------------------------------------------------------

