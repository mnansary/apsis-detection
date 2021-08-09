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


from .render import renderMemoHead,renderMemoTable,renderMemoBottom
from .word import createHandwritenWords
from .utils import placeWordOnMask
from .memo import PAD, Placement,rand_hw_word

#------------------------------------
#  placement
#-----------------------------------

def memo_join(ds,language,iden):
    '''
        joins a memo segments
    '''
    place=Placement()
    head_img,head_print,head_hw,_,iden=renderMemoHead(ds,language,iden)
    table_img,table_print,table_hw,_,iden=renderMemoTable(ds,language,iden)
    bottom_img,bottom_print,bottom_hw,_,iden=renderMemoBottom(ds,language,iden)
    
    # resource
    if language=="bangla":
        g_df     =ds.bangla.graphemes.df 
        n_df     =ds.bangla.numbers.df 
    else:
        g_df     =ds.english.graphemes.df 
        n_df     =ds.english.numbers.df 
    sdf         =  ds.common.symbols.df
    nsdf        =  pd.concat([n_df,sdf],ignore_index=True)
    gsdf        =  pd.concat([g_df,sdf],ignore_index=True)
    adf         =  pd.concat([n_df,g_df,sdf],ignore_index=True)
    noise_signs =  [img_path for img_path in glob(os.path.join(ds.common.noise.sign,"*.bmp"))]
    ## place heads
    region_values=sorted(np.unique(head_hw))[1:]
    max_regs=len(region_values)
    if max_regs<place.head_min:
        place.head_min=max_regs
    len_regs=random.randint(place.head_min,max_regs)
    
    # placement capacities
    ## place heads
    region_values=sorted(np.unique(head_hw))[1:]
    max_regs=len(region_values)
    if max_regs<place.head_min:
        place.head_min=max_regs
    len_regs=random.randint(place.head_min,max_regs)
    
    hw_mask_head=np.zeros_like(head_hw)
    for i in range(len_regs):
        reg_val=random.choice(region_values)
        region_values.remove(reg_val)
        if i==0:
            df=nsdf
            comps=rand_hw_word(nsdf,place.min_word_len,place.max_word_len)
        else:
            df=random.choice([g_df,gsdf,adf])
            comps=rand_hw_word(df,place.min_word_len,place.max_word_len)
        word,labels,iden=createHandwritenWords(iden,df,comps,PAD,place.comp_dim)
        # words
        hw_mask_head=placeWordOnMask(word,head_hw,reg_val,hw_mask_head,cmp_reg=True)

    ## place table
    region_values=sorted(np.unique(table_hw))[1:]
    max_regs=len(region_values)
    if max_regs<place.head_min:
        place.table_min =max_regs
    len_regs=random.randint(place.table_min,max_regs)
    
    hw_mask_table=np.zeros_like(table_hw)
    for i in range(len_regs):
        reg_val=random.choice(region_values)
        region_values.remove(reg_val)
        df=random.choice([n_df,nsdf])
        comps=rand_hw_word(df,place.min_num_len,place.max_num_len)
        word,labels,iden=createHandwritenWords(iden,df,comps,PAD,place.comp_dim)
        # words
        hw_mask_table=placeWordOnMask(word,table_hw,reg_val,hw_mask_table,ext_reg=True)
    ## place bottom
    noise_num=random.choice([1,2])
    region_values=sorted(np.unique(bottom_hw))[1:]
    hw_mask_bottom=np.zeros_like(bottom_hw)
    for i in range(noise_num):
        word=cv2.imread(random.choice(noise_signs),0)
        reg_val=region_values[i]
        hw_mask_bottom=placeWordOnMask(word,bottom_hw,reg_val,hw_mask_bottom,ext_reg=True)

    # joining
    w_max=0
    padded_imgs=[]
    padded_prints=[]
    padded_hws=[]
    for img in [head_img,table_img,bottom_img]:
        h,w=img.shape
        if w>w_max:
            w_max=w
    for idx,imgs in enumerate([[head_img,table_img,bottom_img],
                 [head_print,table_print,bottom_print],
                 [hw_mask_head,hw_mask_table,hw_mask_bottom]]):
        for img in imgs:
            h,w=img.shape
            if w<w_max:
                _pad=np.zeros((h,w_max-w))
                if random.choice([0,1])==1:
                    img=np.concatenate([img,_pad],axis=1)
                else:
                    img=np.concatenate([_pad,img],axis=1)
            if idx==0:
                padded_imgs.append(img)
            if idx==1:
                padded_prints.append(img)
            else:
                padded_hws.append(img)

    return np.concatenate(padded_imgs,axis=0),np.concatenate(padded_prints,axis=0),np.concatenate(padded_hws,axis=0)    
                
