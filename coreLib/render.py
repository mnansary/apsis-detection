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
from coreLib.config import config
from coreLib.word import create_word
#--------------------
# helpers
#--------------------
def padPage(img):
    '''
        pads a page image to proper dimensions
    '''
    h,w=img.shape 
    if h>config.back_dim:
        # resize height
        height=config.back_dim
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad width
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
    else:
        _type=random.choice(["top","bottom","middle"])
        if _type in ["top","bottom"]:
            pad_height=config.back_dim-h
            pad     =np.zeros((pad_height,config.back_dim))
            if _type=="top":
                img=np.concatenate([img,pad],axis=0)
            else:
                img=np.concatenate([pad,img],axis=0)
        else:
            # pad heights
            top_pad_height =(config.back_dim-h)//2
            bot_pad_height=config.back_dim-h-top_pad_height
            # pads
            top_pad =np.zeros((top_pad_height,w),dtype=np.int64)
            bot_pad=np.zeros((bot_pad_height,w),dtype=np.int64)
            # pad
            img =np.concatenate([top_pad,img,bot_pad],axis=0)
    # for error avoidance
    img=cv2.resize(img,(config.back_dim,config.back_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img


def processLine(img):
    '''
        fixes a line image 
        args:
            img        :  concatenated line images
    '''
    h,w=img.shape 
    if w>config.back_dim:
        width=config.back_dim-random.randint(0,config.back_margin)
        # resize
        height= int(width* h/w) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
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
    return img 

def randColor():
    '''
        generates random color
    '''
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

#------------------------
# background
#------------------------

def backgroundGenerator(ds,dim=(1024,1024)):
    '''
        generates random background
        args:
            ds   : dataset object
            dim  : the dimension for background
    '''
    # collect image paths
    _paths=[img_path for img_path in tqdm(glob(os.path.join(ds.common.background,"*.*")))]
    while True:
        _type=random.choice(["single","double","comb"])
        if _type=="single":
            img=cv2.imread(random.choice(_paths))
            img=cv2.resize(img,dim)
            yield img
        elif _type=="double":
            imgs=[]
            img_paths= random.sample(_paths, 2)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                img=cv2.resize(img,dim)
                imgs.append(img)
            # randomly concat
            img=np.concatenate(imgs,axis=random.choice([0,1]))
            img=cv2.resize(img,dim)
            yield img
        else:
            imgs=[]
            img_paths= random.sample(_paths, 4)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                img=cv2.resize(img,dim)
                imgs.append(img)
            seg1=imgs[:2]
            seg2=imgs[2:]
            seg1=np.concatenate(seg1,axis=0)
            seg2=np.concatenate(seg2,axis=0)
            img=np.concatenate([seg1,seg2],axis=1)
            img=cv2.resize(img,dim)
            yield img

#--------------------
# main
#--------------------
def createSceneImage(ds,iden=3):
    '''
        creates a scene image
        args:
            ds  :  the dataset object
            iden:  starting iden for marking
    '''
    iden=iden
    labels=[]
    page_parts=[]
    # select number of lines in an image
    num_lines=random.randint(config.min_num_lines,config.max_num_lines)
    for _ in range(num_lines):
        line_parts=[]
        line_labels=[]
        # select number of words
        num_words=random.randint(config.min_num_words,config.max_num_words)
        for _ in range(num_words):
            img,label,iden=create_word(iden=iden,
                            source_type="bangla",
                            data_type=random.choice(["handwritten","printed"]),
                            comp_type=random.choice(["number","grapheme"]),
                            ds=ds,
                            use_dict=random.choice([True,False]),
                            )
            line_labels.append(label)
            line_parts.append(img)


        # create the line image
        line_img=np.concatenate(line_parts,axis=1)
        line_img=processLine(line_img)
        # the page lines
        page_parts.append(line_img)
        labels.append(line_labels)
    
    
    '''
        single entry to ensure non-zero image
    '''
    
    # Explicit Entry
    line_parts=[]
    line_labels=[]
    img,label,iden=create_word(iden=iden,
                    source_type="bangla",
                    data_type=random.choice(["handwritten","printed"]),
                    comp_type=random.choice(["number","grapheme"]),
                    ds=ds,
                    use_dict=random.choice([True,False]),
                    )
    line_labels.append(label)
    

    line_img=processLine(img)
    # the page lines
    page_parts.append(line_img)
    labels.append(line_labels)
    
    '''
        single entry to ensure non-zero image
    '''
    
    
    paded_parts=[]
    for lidx,line_img in enumerate(page_parts):
        if line_img.shape[0]>=config.min_line_height:
            # pad lines 
            pad_height=random.randint(config.vert_min_space,config.vert_max_space)
            pad     =np.zeros((pad_height,config.back_dim))
            line_img=np.concatenate([line_img,pad],axis=0)
            paded_parts.append(line_img)
        else:
            labels[lidx]=None
    # page img
    page=np.concatenate(paded_parts,axis=0)
    page=padPage(page)
    # eliminate very small noises
    labels=[label for label in labels if label is not None]
    return page,labels