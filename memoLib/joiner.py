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
from numpy.lib.function_base import place
import pandas as pd
from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont
import matplotlib.pyplot as plt 

from .render import renderMemoHead,renderMemoTable,renderMemoBottom
from .word import createHandwritenWords
from .utils import padToFixedHeightWidth, placeWordOnMask
from .memo import PAD, Placement,rand_hw_word

#------------------------------------
#  placement
#-----------------------------------

def memo_placement(ds,language,pad_dim=10):
    '''
        joins a memo segments
    '''
    # resource: can be more optimized 
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
    
    # extract images and regions
    table_img,table_print,table_reg      =   renderMemoTable(ds,language)
    _,tbm=table_img.shape
    head_img,head_print,head_reg         =   renderMemoHead(ds,language,tbm)
    bottom_img,bottom_print,bottom_reg   =   renderMemoBottom(ds,language,tbm)
    
    

    place=Placement()
    ## place heads
    region_values=sorted(np.unique(head_reg))[1:]
    region_values=[int(v) for v in region_values]
    max_regs=len(region_values)
    if max_regs<place.head_min:
        place.head_min=max_regs
    len_regs=random.randint(place.head_min,max_regs)
    
    head_hw=np.zeros_like(head_reg)
    for i in range(len_regs):
        reg_val=random.choice(region_values)
        region_values.remove(reg_val)
        if i==0:
            df=nsdf
            comps=rand_hw_word(nsdf,place.min_word_len,place.max_word_len)
        else:
            df=random.choice([g_df,gsdf,adf])
            comps=rand_hw_word(df,place.min_word_len,place.max_word_len)
        word=createHandwritenWords(df,comps,PAD,place.comp_dim)
        # words
        head_hw=placeWordOnMask(word,head_reg,reg_val,head_hw,ext_reg=True,fill=False,ext=(30,50))
    
    ## place table
    region_values=sorted(np.unique(table_reg))[1:]
    region_values=[int(v) for v in region_values]
    max_regs=len(region_values)
    if max_regs<place.table_min:
        place.table_min=max_regs
    len_regs=random.randint(place.table_min,place.table_min*2)
    
    table_hw=np.zeros_like(table_reg)
    for i in range(len_regs):
        reg_val=random.choice(region_values)
        region_values.remove(reg_val)
        df=random.choice([n_df,nsdf])
        comps=rand_hw_word(df,place.min_num_len,place.max_num_len)
        word=createHandwritenWords(df,comps,PAD,place.comp_dim)
        # words
        table_hw=placeWordOnMask(word,table_reg,reg_val,table_hw,ext_reg=True,fill=True)
    
    ## place bottom
    noise_num=random.choice([1,2])
    region_values=sorted(np.unique(bottom_reg))[1:]
    bottom_hw=np.zeros_like(bottom_reg)
    for i in range(noise_num):
        word=cv2.imread(random.choice(noise_signs),0)
        word=255-word
        word[word>0]=1
        reg_val=region_values[i]
        bottom_hw=placeWordOnMask(word,bottom_reg,reg_val,bottom_hw,ext_reg=True,fill=False,ext=(30,50))
    
    # construct print img
    _,w_table=table_img.shape
    _,w_head =head_img.shape
    _,w_bottoom =bottom_img.shape
    max_w=max(w_table,w_head,w_bottoom)

    # head
    head_img=padToFixedHeightWidth(head_img,head_img.shape[0]+2*pad_dim,max_w)
    head_print=padToFixedHeightWidth(head_print,head_img.shape[0]+2*pad_dim,max_w)
    head_hw=padToFixedHeightWidth(head_hw,head_img.shape[0]+2*pad_dim,max_w)
    # bottom
    bottom_img=padToFixedHeightWidth(bottom_img,bottom_img.shape[0]+2*pad_dim,max_w)
    bottom_print=padToFixedHeightWidth(bottom_print,bottom_img.shape[0]+2*pad_dim,max_w)
    bottom_hw=padToFixedHeightWidth(bottom_hw,bottom_img.shape[0]+2*pad_dim,max_w)
    # table
    table_img=padToFixedHeightWidth(table_img,table_img.shape[0]+2*pad_dim,max_w)
    table_print=padToFixedHeightWidth(table_print,table_img.shape[0]+2*pad_dim,max_w)
    table_hw=padToFixedHeightWidth(table_hw,table_img.shape[0]+2*pad_dim,max_w)
    #
    memo_img=np.concatenate([head_img,table_img,bottom_img],axis=0)
    memo_raw_hw=np.concatenate([head_hw,table_hw,bottom_hw],axis=0)
    #memo_img[memo_raw_hw==1]=2
    
    memo_print=np.concatenate([head_print,table_print,np.zeros_like(bottom_img)],axis=0)
    memo_hw=np.concatenate([head_hw,table_hw,np.zeros_like(bottom_hw)],axis=0)
    
    plt.imshow(memo_img)
    plt.show()
    plt.imshow(memo_print)
    plt.show()
    plt.imshow(memo_hw)
    plt.show()
    plt.imshow(memo_raw_hw)
    plt.show()
    
    

    # # joining
    # w_max=0
    # padded_imgs=[]
    # padded_prints=[]
    # padded_hws=[]
    # for img in [head_img,table_img,bottom_img]:
    #     h,w=img.shape
    #     if w>w_max:
    #         w_max=w
    # for idx,imgs in enumerate([[head_img,table_img,bottom_img],
    #              [head_print,table_print,bottom_print],
    #              [hw_mask_head,hw_mask_table,hw_mask_bottom]]):
    #     for img in imgs:
    #         h,w=img.shape
    #         if w<w_max:
    #             _pad=np.zeros((h,w_max-w))
    #             if random.choice([0,1])==1:
    #                 img=np.concatenate([img,_pad],axis=1)
    #             else:
    #                 img=np.concatenate([_pad,img],axis=1)
    #         if idx==0:
    #             padded_imgs.append(img)
    #         if idx==1:
    #             padded_prints.append(img)
    #         else:
    #             padded_hws.append(img)

    # return np.concatenate(padded_imgs,axis=0),np.concatenate(padded_prints,axis=0),np.concatenate(padded_hws,axis=0)    
                
