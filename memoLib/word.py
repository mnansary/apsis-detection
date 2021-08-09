# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import regex 
import numpy as np 
import cv2
import os
from glob import glob 
import PIL.Image,PIL.ImageDraw,PIL.ImageFont
import random
import pandas as pd 

# from .config import config
from .utils import stripPads

#-----------------------------------
# line image
#----------------------------------
def handleExtensions(ext,font,max_width):
    '''
        creates/ adds extensions to lines
    '''
    width = font.getsize(ext)[0]
    
    # draw
    image = PIL.Image.new(mode='L', size=font.getsize(ext))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=ext, fill=1, font=font)
    num_ext=max_width//width
    if num_ext>1:
        ext_img=[np.array(image) for _ in range(max_width//width)]
        ext_img=np.concatenate(ext_img,axis=1)
        return ext_img
    else:
        return None

def createPrintedLine(line,font):
    '''
        creates printed word image
        args:
            line           :       the string
            font           :       the desired font
            
        returns:
            img     :       printed line image
            
    '''
    # draw
    image = PIL.Image.new(mode='L', size=font.getsize(line))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=line, fill=1, font=font)
    return np.array(image)

    
#-----------------------------------
# hw image
#----------------------------------
def createHandwritenWords(df,
                         comps,
                         pad,
                         comp_dim):
    '''
        creates handwriten word image
        args:
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components 
            pad     :       pad class:
                                no_pad_dim
                                single_pad_dim
                                double_pad_dim
                                top
                                bot
            comp_dim:       component dimension 
        returns:
            img     :       marked word image
            
    '''
    comps=[str(comp) for comp in comps]
    # select a height
    height=comp_dim
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]

    # alignment of component
    ## flags
    tp=False
    bp=False
    comp_heights=["" for _ in comps]
    for idx,comp in enumerate(comps):
        if any(te.strip() in comp for te in pad.top):
            comp_heights[idx]+="t"
            tp=True
        if any(be in comp for be in pad.bot):
            comp_heights[idx]+="b"
            bp=True


    imgs=[]
    for cidx,comp in enumerate(comps):
        c_df=df.loc[df.label==comp]
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=c_df.iloc[idx,2] 
        # read image
        img=cv2.imread(img_path,0)

        # resize
        hf=comp_heights[cidx]
        if hf=="":
            img=cv2.resize(img,pad.no_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)
        elif hf=="t":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)

        elif hf=="b":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
        elif hf=="bt" or hf=="tb":
            img=cv2.resize(img,pad.double_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        
        
        
        # mark image
        img=255-img
        img[img>0]=1
        imgs.append(img)
        
    img=np.concatenate(imgs,axis=1)
    h,w=img.shape 
    width= int(height* w/h) 
    img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img