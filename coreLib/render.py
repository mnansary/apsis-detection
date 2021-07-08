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
from .word import create_word
from .utils import processLine,padPage,randColor 
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
# page
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
            img,label,iden=create_word( iden=iden,
                                        ds=ds,
                                        source_type=random.choice(config.data.sources),
                                        data_type=random.choice(config.data.formats),
                                        comp_type=random.choice(config.data.components), 
                                        use_dict=random.choice([True,False]))
            line_labels.append(label)
            line_parts.append(img)


        # create the line image
        line_img=np.concatenate(line_parts,axis=1)
        line_img=processLine(line_img)
        # the page lines
        page_parts.append(line_img)
        labels.append(line_labels)
    
    paded_parts=[]
    for lidx,line_img in enumerate(page_parts):
        # pad lines 
        pad_height=random.randint(config.vert_min_space,config.vert_max_space)
        pad     =np.zeros((pad_height,config.back_dim))
        line_img=np.concatenate([line_img,pad],axis=0)
        paded_parts.append(line_img)

    # page img
    page=np.concatenate(paded_parts,axis=0)
    page=padPage(page)
    
    return page,labels


#--------------------
# data
#--------------------
def createImageData(backgen,page,labels):
    '''
        creates a proper image to save 
        args:
            backgen :   background generator
            page    :   the page image
            labels  :   the labels of the page
    '''
    back=next(backgen)

    for line_label in labels:
        # random choice for color distribution
        _colType=random.choice(["inline","different"])
        if _colType=="inline":
            line_col=randColor()
        else:
            line_col=None
        for label in line_label:
            # format color space
            if line_col is None:
                col=randColor()
            else:
                col=line_col
            # place colors
            for k,v in label.items():
                if v!=' ':
                    back[page==k]=col
    return back
