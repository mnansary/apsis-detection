# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from .config import config
from .wordmaps import create_word
from .utils import draw_random_noise, randColor, random_exec

def padMaps(img,hmap,lmap):
    '''
        pads a page image to proper dimensions
    '''
    h,w=img.shape 
    if h>config.back_dim:
        # resize height
        height=config.back_dim
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        hmap=cv2.resize(hmap,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        lmap=cv2.resize(lmap,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        # pad width
        # mandatory check
        h,w=img.shape 
        # pad widths
        left_pad_width =random.randint(0,(config.back_dim-w))
        right_pad_width=config.back_dim-w-left_pad_width
        # pads
        left_pad =np.zeros((h,left_pad_width))
        right_pad=np.zeros((h,right_pad_width))
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
        hmap =np.concatenate([left_pad,hmap,right_pad],axis=1)
        lmap =np.concatenate([left_pad,lmap,right_pad],axis=1)
    else:
        _type=random.choice(["top","bottom","middle"])
        if _type in ["top","bottom"]:
            pad_height=config.back_dim-h
            pad     =np.zeros((pad_height,config.back_dim))
            if _type=="top":
                img=np.concatenate([img,pad],axis=0)
                hmap=np.concatenate([hmap,pad],axis=0)
                lmap=np.concatenate([lmap,pad],axis=0)
            else:
                img=np.concatenate([pad,img],axis=0)
                hmap=np.concatenate([pad,hmap],axis=0)
                lmap=np.concatenate([pad,lmap],axis=0)
        else:
            # pad heights
            top_pad_height =(config.back_dim-h)//2
            bot_pad_height=config.back_dim-h-top_pad_height
            # pads
            top_pad =np.zeros((top_pad_height,w))
            bot_pad=np.zeros((bot_pad_height,w))
            # pad
            img =np.concatenate([top_pad,img,bot_pad],axis=0)
            hmap =np.concatenate([top_pad,hmap,bot_pad],axis=0)
            lmap =np.concatenate([top_pad,lmap,bot_pad],axis=0)
    # for error avoidance
    img=cv2.resize(img,(config.back_dim,config.back_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    hmap=cv2.resize(hmap,(config.back_dim,config.back_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    lmap=cv2.resize(lmap,(config.back_dim,config.back_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    return img,hmap,lmap


def processLineMaps(img,hmap,lmap):
    '''
        fixes a line image 
        args:
            img         :  concatenated line images
            hmap        :  concatenated heat map images
            lmap        :  concatenated link map images
    '''
    h,w=img.shape 
    if w>config.back_dim:
        width=config.back_dim-random.randint(0,300)
        # resize
        height= int(width* h/w) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        hmap=cv2.resize(hmap,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        lmap=cv2.resize(lmap,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # mandatory check
    h,w=img.shape 
    # pad widths
    left_pad_width =random.randint(0,(config.back_dim-w))
    right_pad_width=config.back_dim-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width),dtype=np.int64)
    right_pad=np.zeros((h,right_pad_width),dtype=np.int64)
    # pad
    img =np.concatenate([left_pad,img,right_pad],axis=1)
    hmap =np.concatenate([left_pad,hmap,right_pad],axis=1)
    lmap =np.concatenate([left_pad,lmap,right_pad],axis=1)
    
    return img,hmap,lmap 


#--------------------
# page
#--------------------
def createSceneMaps(ds,gmap,backgen):
    '''
        creates a scene image
        args:
            ds      :  the dataset object
            gmap    :       gaussian heatmap
            backgen :   background generator
    '''
    word_iden=1
    page_imgs=[]
    page_hmaps=[]
    page_lmaps=[]
    
    # select number of lines in an image
    num_lines=random.randint(config.min_num_lines,config.max_num_lines)
    for _ in range(num_lines):
        line_imgs=[]
        line_hmaps=[]
        line_lmaps=[]
        
        # select number of words
        num_words=random.randint(config.min_num_words,config.max_num_words)
        for _ in range(num_words):
            img,hmap,lmap=create_word( gmap=gmap,
                                        word_iden=word_iden,
                                        source_type=random.choice(config.data.sources),
                                        data_type=random.choice(config.data.formats),
                                        comp_type=random.choice(config.data.components), 
                                        ds=ds,
                                        use_dict=random.choice([True,False]))
            line_imgs.append(img)
            line_hmaps.append(hmap)
            line_lmaps.append(lmap)
            word_iden+=1
            
        
        # reform
        rline_imgs=[]
        rline_hmaps=[]
        rline_lmaps=[]
        max_h=0
        # find max height
        for line_img in line_imgs:
            max_h=max(max_h,line_img.shape[0])
        
        # reform
        for img,hmap,lmap in zip(line_imgs,line_hmaps,line_lmaps):
            h,w=img.shape 
            width= int(max_h* w/h) 
            img=cv2.resize(img,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            hmap=cv2.resize(hmap,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            lmap=cv2.resize(lmap,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            
            rline_imgs.append(img)
            rline_hmaps.append(hmap)
            rline_lmaps.append(lmap)
            

        # create the line image
        line_img=np.concatenate(rline_imgs,axis=1)
        line_hmap=np.concatenate(rline_hmaps,axis=1)
        line_lmap=np.concatenate(rline_lmaps,axis=1)
        

        line_img,line_hmap,line_lmap=processLineMaps(line_img,line_hmap,line_lmap)
        # the page lines
        page_imgs.append(line_img)
        page_hmaps.append(line_hmap)
        page_lmaps.append(line_lmap)
        
    imgs=[]
    hmaps=[]
    lmaps=[]
    for img,hmap,lmap in zip(page_imgs,page_hmaps,page_lmaps):
        # pad lines 
        pad_height=random.randint(config.vert_min_space,config.vert_max_space)
        pad     =np.zeros((pad_height,config.back_dim))
        img=np.concatenate([img,pad],axis=0)
        hmap=np.concatenate([hmap,pad],axis=0)
        lmap=np.concatenate([lmap,pad],axis=0)
        
        imgs.append(img)
        hmaps.append(hmap)
        lmaps.append(lmap)
        

    # page data img
    img=np.concatenate(imgs,axis=0)
    hmap=np.concatenate(hmaps,axis=0)
    lmap=np.concatenate(lmaps,axis=0)
    img,hmap,lmap=padMaps(img,hmap,lmap)

    # scene
    back=next(backgen)
    vals=[v for v in np.unique(img) if v>0]

    for v in vals:
        col=randColor()
        back[img==v]=col
    
        
    return back,hmap,lmap

def createNoisyMaps(ds,gmap):
    '''
        creates a scene image
        args:
            ds      :  the dataset object
            gmap    :       gaussian heatmap
    '''
    word_iden=1
    page_imgs=[]
    page_hmaps=[]
    page_lmaps=[]
    
    # select number of lines in an image
    num_lines=random.randint(config.min_num_lines,config.max_num_lines)
    for _ in range(num_lines):
        line_imgs=[]
        line_hmaps=[]
        line_lmaps=[]
        
        # select number of words
        num_words=random.randint(config.min_num_words,config.max_num_words)
        for _ in range(num_words):
            img,hmap,lmap=create_word( gmap=gmap,
                                        word_iden=word_iden,
                                        source_type=random.choice(config.data.sources),
                                        data_type=random.choice(config.data.formats),
                                        comp_type=random.choice(config.data.components), 
                                        ds=ds,
                                        use_dict=random.choice([True,False]))
            line_imgs.append(img)
            line_hmaps.append(hmap)
            line_lmaps.append(lmap)
            word_iden+=1
            
        
        # reform
        rline_imgs=[]
        rline_hmaps=[]
        rline_lmaps=[]
        max_h=0
        # find max height
        for line_img in line_imgs:
            max_h=max(max_h,line_img.shape[0])
        
        # reform
        for img,hmap,lmap in zip(line_imgs,line_hmaps,line_lmaps):
            h,w=img.shape 
            width= int(max_h* w/h) 
            img=cv2.resize(img,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            hmap=cv2.resize(hmap,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            lmap=cv2.resize(lmap,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            
            rline_imgs.append(img)
            rline_hmaps.append(hmap)
            rline_lmaps.append(lmap)
            

        # create the line image
        line_img=np.concatenate(rline_imgs,axis=1)
        line_hmap=np.concatenate(rline_hmaps,axis=1)
        line_lmap=np.concatenate(rline_lmaps,axis=1)
        

        line_img,line_hmap,line_lmap=processLineMaps(line_img,line_hmap,line_lmap)
        # the page lines
        page_imgs.append(line_img)
        page_hmaps.append(line_hmap)
        page_lmaps.append(line_lmap)
        
    imgs=[]
    hmaps=[]
    lmaps=[]
    for img,hmap,lmap in zip(page_imgs,page_hmaps,page_lmaps):
        # pad lines 
        pad_height=random.randint(config.vert_min_space,config.vert_max_space)
        pad     =np.zeros((pad_height,config.back_dim))
        img=np.concatenate([img,pad],axis=0)
        hmap=np.concatenate([hmap,pad],axis=0)
        lmap=np.concatenate([lmap,pad],axis=0)
        
        imgs.append(img)
        hmaps.append(hmap)
        lmaps.append(lmap)
        

    # page data img
    img=np.concatenate(imgs,axis=0)
    hmap=np.concatenate(hmaps,axis=0)
    lmap=np.concatenate(lmaps,axis=0)
    img,hmap,lmap=padMaps(img,hmap,lmap)

    # scene
    h,w=img.shape
    back=np.ones((h,w,3))*255
    back=back.astype("uint8")
    vals=[v for v in np.unique(img) if v>0]

    for v in vals:
        col=randColor()
        back[img==v]=(0,0,0)
    if random_exec():
        back=draw_random_noise(back,img)
        
    return back,hmap,lmap
