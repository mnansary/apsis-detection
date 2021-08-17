# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary
'''
#----------------------------
# imports
#----------------------------
import numpy as np
import random
import cv2
import os
import cv2
import string

import pandas as pd
from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont
import matplotlib.pyplot as plt 

from .render_head import renderMemoHead
from .render_table import renderMemoTable
from .render_bottom import renderMemoBottom
from .utils import padToFixedHeightWidth
#------------------------------------
#  placement
#-----------------------------------

def create_memo_data(ds,language,img_height=1024):
    '''
        joins a memo segments
    '''
    # extract images and regions
    table_img,table_cmap,table_wmap      =   renderMemoTable(ds,language)
    _,tbm,_=table_img.shape
    head_img,head_cmap,head_wmap         =   renderMemoHead(ds,language,tbm)
    bottom_img,bottom_cmap,bottom_wmap   =   renderMemoBottom(ds,language,tbm)
    
    # maps
    img =np.concatenate([head_img,table_img,bottom_img],axis=0)
    cmap=np.concatenate([head_cmap,table_cmap,bottom_cmap],axis=0)
    wmap=np.concatenate([head_wmap,table_wmap,bottom_wmap],axis=0)
    # resizing
    h,w,d=img.shape
    w_new=int(img_height*w/h)
    img =cv2.resize(img,(w_new,img_height))
    cmap=cv2.resize(cmap,(w_new,img_height),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
    wmap=cv2.resize(wmap,(w_new,img_height),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
    img     =   img.astype("uint8")
    img     =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur    =   cv2.GaussianBlur(img,(5,5),0)
    _,img   =   cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=np.ones_like(img)
    
    img     =   padToFixedHeightWidth(img,img_height,img_height)
    cmap    =   padToFixedHeightWidth(cmap,img_height,img_height)
    wmap    =   padToFixedHeightWidth(wmap,img_height,img_height)
    mask    =   padToFixedHeightWidth(mask,img_height,img_height)
    img[mask==0]=255
    return img,cmap,wmap
    