# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary, MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np
import random
import os
import cv2
import string

from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont

from .memo import rand_head,Head,LineSection,LineWithExtension
from .word import createPrintedLine
from .utils import padLineImg

#----------------------------
# render capacity: toolset
#----------------------------
def renderFontMaps(LineSection,font_path):
    '''
        renders a font map
    '''
    maps={}
    sizes=LineSection.font_sizes_big+LineSection.font_sizes_mid
    for size in sizes:
        maps[str(size)]=PIL.ImageFont.truetype(font_path, size=size)
    return maps

#----------------------------
# render capacity: memo head
#----------------------------
def renderMemoHead(ds,language,iden):


    """
        @function author:        
        Create image of top part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
    head=rand_head(graphemes,numbers,Head,LineSection,LineWithExtension)
    maps=renderFontMaps(LineSection,random.choice(font_paths))
    
    
    h_max=0
    w_max=0
    line_images=[]
    line_labels=[]
    # create line sections
    for line_data in head.line_sections:
        assert len(line_data)==1
        data=line_data[0]
        img,labels,iden=createPrintedLine(iden=iden,words=data["words"],font=maps[str(data["font_size"])],font_size=data["font_size"])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        line_images.append(img)
        line_labels+=labels
    
    line_images=[padLineImg(line_img,h_max,w_max) for line_img in line_images]
    
    # create double ext sections
    for data in head.double_exts:
        assert len(data)==2
        img1,labels1,iden=createPrintedLine(iden=iden,words=data[0]["words"],font=maps[str(data[0]["font_size"])],font_size=data[0]["font_size"])
        # add ext
        h1,w1=img1.shape
        ext_w=w_max//2-w1
        ext=np.ones((h1,ext_w))*iden
        labels1.append({f"{iden}":"ext"})
        iden+=1        
        img1=np.concatenate([img1,ext],axis=1)
        
        img2,labels2,iden=createPrintedLine(iden=iden,words=data[0]["words"],font=maps[str(data[0]["font_size"])],font_size=data[0]["font_size"])
        # add ext
        h2,w2=img2.shape
        ext_w=w_max//2-w2
        ext=np.ones((h2,ext_w))*iden 
        labels2.append({f"{iden}":"ext"})
        iden+=1
        img2=np.concatenate([img2,ext],axis=1)
        
        img=np.concatenate([img1,img2],axis=1)
        # correction
        h,w=img.shape
        if w<w_max:
            pad=np.zeros((h,w_max-w))
            img=np.concatenate([img,pad],axis=1)
        # append
        line_images.append(img)
        line_labels+=labels1+labels2
        

    # create single ext sections
    for line_data in head.single_exts:
        assert len(line_data)==1
        data=line_data[0]
        img,labels,iden=createPrintedLine(iden=iden,words=data["words"],font=maps[str(data["font_size"])],font_size=data["font_size"])
        # add ext
        h,w=img.shape
        ext_w=w_max-w
        ext=np.ones((h,ext_w))*iden
        labels.append({f"{iden}":"ext"})
        iden+=1
        
        img=np.concatenate([img,ext],axis=1)
        # append
        line_images.append(img)
        line_labels+=labels
    
    memo_head=np.concatenate(line_images,axis=0)
    return memo_head,line_labels,iden

#----------------------------
# render capacity: table 
#----------------------------
