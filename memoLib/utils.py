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

def padToFixedHeightWidth(img,h_max,w_max):
    '''
        pads an image to fixed height and width
    '''
    # shape
    h,w=img.shape
    # pad widths
    left_pad_width =(w_max-w)//2
    # print(left_pad_width)
    right_pad_width=w_max-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width))
    right_pad=np.zeros((h,right_pad_width))
    # pad
    img =np.concatenate([left_pad,img,right_pad],axis=1)
    
    # shape
    h,w=img.shape
    # pad heights
    top_pad_height =(h_max-h)//2
    bot_pad_height=h_max-h-top_pad_height
    # pads
    top_pad =np.zeros((top_pad_height,w))
    bot_pad=np.zeros((bot_pad_height,w))
    # pad
    img =np.concatenate([top_pad,img,bot_pad],axis=0)
    return img

def padAllAround(img,pad_dim):
    '''
        pads all around the image
    '''
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

def padToFixImgWidth(img,width):
    '''
        fix image based purely by width
    '''
    h,w=img.shape
    # case 1: w< width
    if w<width:
        pad_w=width-w
        if pad_w%2==0:
            pad =np.zeros((h,pad_w//2))
            img=np.concatenate([pad,img,pad],axis=1)
        else:
            pad=np.zeros((h,pad_w))
            if random.choice([1,0])==1:
                img=np.concatenate([img,pad],axis=1)
            else:
                img=np.concatenate([pad,img],axis=1)
    else:
        h_needed=int(width* h/w) 
        img = cv2.resize(img, (width,h_needed), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        pad_h=h-h_needed
        if pad_h%2==0:
            pad =np.zeros((pad_h//2,width))
            img=np.concatenate([pad,img,pad],axis=0)
        else:
            pad=np.zeros((pad_h,width))
            if random.choice([1,0])==1:
                img=np.concatenate([img,pad],axis=0)
            else:
                img=np.concatenate([pad,img],axis=0)
    return img 
#---------------------------------------------------------------
def placeWordOnMask(word,labeled_img,region_value,mask,ext_reg=False,fill=False):
    '''
        @author
        places a specific image on a given background at a specific location
        args:
            word               :   greyscale image to place
            labeled_img        :   labeled image to place the image
            region_value       :   the specific value of the labled region
            mask               :   placement mask
            ext_reg            :   extend the region to place
            fill
        return:
            mak :   mask image after placing 'img'
    '''
    idx=np.where(labeled_img==region_value)
    # region
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    if ext_reg:
        h_li,w_li=labeled_img.shape
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
    
    if fill:
        # resize image    
        h_max = abs(y_max-y_min)
        w_max = abs(x_max-x_min)
        word = cv2.resize(word, (w_max,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    else:# unstable NOW    
        # resize image    
        h_max = abs(y_max-y_min)
        w_max = abs(x_max-x_min)
        h,w=word.shape
        w_needed=int(h_max* w/h) 
        word = cv2.resize(word, (w_needed,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # fix padding
        word=padToFixImgWidth(word,w_max)    
    # place on mask
    mask[y_min:y_max,x_min:x_max]=word
    return mask
#---------------------------------------------------------------

