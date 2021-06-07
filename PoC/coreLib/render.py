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
from tqdm.auto import tqdm
from coreLib.config import config
from coreLib.word import create_word
#--------------------
# helpers
#--------------------

def padLineLeftRight(max_line_width,line_img):
    '''
        pads an image left and right
        args:
            max_line_width  : width of the max line length
            line_img        : image to pad
    '''
    # shape
    h,w=line_img.shape
    # pad widths
    left_pad_width =random.randint(0,(max_line_width-w))
    right_pad_width=max_line_width-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width),dtype=np.int64)
    right_pad=np.zeros((h,right_pad_width),dtype=np.int64)
    # pad
    line_img =np.concatenate([left_pad,line_img,right_pad],axis=1)
    return line_img

def randColor():
    '''
        generates random color
    '''
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

#------------------------
# background
#------------------------

def backgroundGenerator(ds,dim=(1024,1024),_type=None):
    '''
        generates random background
        args:
            ds   : dataset object
            dim  : the dimension for background
            _type: "single","double","comb" to generate various forms
    '''
    # collect image paths
    _paths=[img_path for img_path in tqdm(glob(os.path.join(ds.common.background,"*.*")))]
    while True:
        if _type is None:
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
        # select number of words
        num_words=random.randint(config.min_num_words,config.max_num_words)
        for _ in range(num_words):
            img,label,iden=create_word(iden=iden,
                            source_type="bangla",
                            data_type=random.choice(["handwritten","handwritten","handwritten","printed"]),
                            comp_type="grapheme",
                            ds=ds,
                            use_dict=random.choice([True,True,False]),
                            )
            labels.append(label)
            line_parts.append(img)


        # create the line image
        line_img=np.concatenate(line_parts,axis=1)
        # the page lines
        page_parts.append(line_img)
    
    
    # find max line width
    max_line_width=0
    for line in page_parts:
        _,w=line.shape
        if w>=max_line_width:
            max_line_width=w
            
    # pad each line to max_width
    paded_parts=[]
    for lidx,line_img in enumerate(page_parts):
        line_img=padLineLeftRight(max_line_width,line_img)
        # top pad for first one
        if lidx==0:
            pad_height_top=random.randint(config.vert_min_space,config.vert_max_space*2)
            pad_top=np.zeros((pad_height_top,max_line_width))
            line_img=np.concatenate([pad_top,line_img],axis=0)
        # pad lines 
        pad_height=random.randint(config.vert_min_space,config.vert_max_space)
        pad     =np.zeros((pad_height,max_line_width))
        line_img=np.concatenate([line_img,pad],axis=0)
        paded_parts.append(line_img)
    # page img
    page=np.concatenate(paded_parts,axis=0)
    h_pad,_=page.shape
    _pad=np.zeros((h_pad,random.randint(config.vert_min_space,config.vert_max_space*2)),dtype=np.int64)
    page=np.concatenate([_pad,page,_pad],axis=1)
    
    return page,labels