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

from .render import renderMemoTable#,renderMemoHead,renderMemoBottom
from .word import createHandwritenWords
from .utils import padToFixedHeightWidth, placeWordOnMask,randColor,rotate_image
from .memo import PAD, Placement,rand_hw_word

#------------------------------------
#  placement
#-----------------------------------

# def create_memo_data(ds,language,pad_dim=10):
#     '''
#         joins a memo segments
#     '''
#     # resource: can be more optimized 
#     if language=="bangla":
#         g_df     =ds.bangla.graphemes.df 
#         n_df     =ds.bangla.numbers.df 
#     else:
#         g_df     =ds.english.graphemes.df 
#         n_df     =ds.english.numbers.df 
#     sdf         =  ds.common.symbols.df
#     nsdf        =  pd.concat([n_df,sdf],ignore_index=True)
#     gsdf        =  pd.concat([g_df,sdf],ignore_index=True)
#     adf         =  pd.concat([n_df,g_df,sdf],ignore_index=True)
#     noise_signs =  [img_path for img_path in glob(os.path.join(ds.common.noise.sign,"*.bmp"))]
    
#     # extract images and regions
#     table_img,table_print,table_reg      =   renderMemoTable(ds,language)
#     _,tbm=table_img.shape
#     head_img,head_print,head_reg         =   renderMemoHead(ds,language,tbm)
#     bottom_img,bottom_print,bottom_reg   =   renderMemoBottom(ds,language,tbm)
    
    

#     place=Placement()
#     ## place heads
#     region_values=sorted(np.unique(head_reg))[1:]
#     region_values=[int(v) for v in region_values]
#     max_regs=len(region_values)
#     if max_regs<place.head_min:
#         place.head_min=max_regs
#     len_regs=random.randint(place.head_min,max_regs)
    
#     head_hw=np.zeros_like(head_reg)
#     for i in range(len_regs):
#         reg_val=random.choice(region_values)
#         region_values.remove(reg_val)
#         if i==0:
#             df=nsdf
#             comps=rand_hw_word(nsdf,place.min_word_len,place.max_word_len)
#         else:
#             df=random.choice([g_df,gsdf,adf])
#             comps=rand_hw_word(df,place.min_word_len,place.max_word_len)
#         word=createHandwritenWords(df,comps,PAD,place.comp_dim)
#         # words
#         head_hw=placeWordOnMask(word,head_reg,reg_val,head_hw,ext_reg=True,fill=False,ext=(10,30))
    
#     ## place table
#     region_values=sorted(np.unique(table_reg))[1:]
#     region_values=[int(v) for v in region_values]
#     max_regs=len(region_values)
#     if max_regs<place.table_min:
#         place.table_min=max_regs
#     len_regs=random.randint(place.table_min,place.table_min*2)
    
#     table_hw=np.zeros_like(table_reg)
#     for i in range(len_regs):
#         reg_val=random.choice(region_values)
#         region_values.remove(reg_val)
#         df=random.choice([n_df,nsdf])
#         comps=rand_hw_word(df,place.min_num_len,place.max_num_len)
#         word=createHandwritenWords(df,comps,PAD,place.comp_dim)
#         # words
#         table_hw=placeWordOnMask(word,table_reg,reg_val,table_hw,ext_reg=True,fill=True)
    
#     ## place bottom
#     noise_num=random.choice([1,2])
#     region_values=sorted(np.unique(bottom_reg))[1:]
#     bottom_hw=np.zeros_like(bottom_reg)
#     for i in range(noise_num):
#         word=cv2.imread(random.choice(noise_signs),0)
#         word=255-word
#         word[word>0]=1
#         reg_val=region_values[i]
#         bottom_hw=placeWordOnMask(word,bottom_reg,reg_val,bottom_hw,ext_reg=True,fill=False,ext=(10,30))
    
#     # construct print img
#     _,w_table=table_img.shape
#     _,w_head =head_img.shape
#     _,w_bottoom =bottom_img.shape
#     max_w=max(w_table,w_head,w_bottoom)

#     # head
#     max_h=head_img.shape[0]+2*pad_dim
#     head_img=padToFixedHeightWidth(head_img,max_h,max_w)
#     head_print=padToFixedHeightWidth(head_print,max_h,max_w)
#     head_hw=padToFixedHeightWidth(head_hw,max_h,max_w)
#     # bottom
#     max_h=bottom_img.shape[0]+2*pad_dim
#     bottom_img=padToFixedHeightWidth(bottom_img,max_h,max_w)
#     bottom_print=padToFixedHeightWidth(bottom_print,max_h,max_w)
#     bottom_hw=padToFixedHeightWidth(bottom_hw,max_h,max_w)
#     # table
#     max_h=table_img.shape[0]+2*pad_dim
#     table_img=padToFixedHeightWidth(table_img,max_h,max_w)
#     table_print=padToFixedHeightWidth(table_print,max_h,max_w)
#     table_hw=padToFixedHeightWidth(table_hw,max_h,max_w)
#     #
#     memo_img=np.concatenate([head_img,table_img,bottom_img],axis=0)
#     memo_raw_hw=np.concatenate([head_hw,table_hw,bottom_hw],axis=0)
    
#     memo_print=np.concatenate([head_print,table_print,np.zeros_like(bottom_print)],axis=0)
#     memo_hw=np.concatenate([head_hw,table_hw,np.zeros_like(bottom_hw)],axis=0)
#     memo_table=np.concatenate([np.zeros_like(head_img),table_img+table_hw,np.zeros_like(bottom_img)],axis=0)

#     h,w=memo_img.shape
#     memo_img[memo_img>0]=255
#     memo_raw_hw[memo_raw_hw>0]=255
#     memo_table[memo_table>0]=255
    
#     memo_data=memo_hw+memo_print
#     memo_data[memo_data>0]=255
    
    
#     memo_3=np.ones((h,w,3))*255
#     memo_3[memo_raw_hw>0]=(0,0,0)
#     # if random.choice([1,0,0,0,0,0])==1:
#     #     col=randColor()
#     # else:
#     #   col=(0,0,0)
#     col=(0,0,0)
#     memo_3[memo_img>0]=col
#     memo_3=memo_3.astype("uint8")
#     return memo_3,memo_print,memo_hw,memo_table

def create_table_data(ds,language,pad_dim=10):
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
    table_img,table_print,table_reg,table_cmap,table_wmap      =   renderMemoTable(ds,language)
    
    

    place=Placement()
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
        word,cmap,wmap=createHandwritenWords(df,comps,PAD,place.comp_dim)
        
        if random.choices(population=[1,0],weights=place.rot_weights,k=1)[0]==1:
            angle=random.randint(place.min_rot,place.max_rot)
            angle=random.choice([angle,-1*angle])
            word=rotate_image(word,angle)
            wmap=rotate_image(wmap,angle)
            cmap=rotate_image(cmap,angle)
            
        ext=random.randint(0,30)
        # words
        table_hw=placeWordOnMask(word,table_reg,reg_val,table_hw,ext_reg=True,fill=True,ext=ext)
        table_cmap=placeWordOnMask(cmap,table_reg,reg_val,table_cmap,ext_reg=True,fill=True,ext=ext)
        table_wmap=placeWordOnMask(wmap,table_reg,reg_val,table_wmap,ext_reg=True,fill=True,ext=ext)
    
    h,w=table_img.shape
    table_img[table_img>0]=255
    table_hw[table_hw>0]=255
    table_print[table_print>0]=255
    
    
    table_3=np.ones((h,w,3))*255
    table_3[table_hw>0]=(0,0,0)
    if random.choice([1,0])==1:
        col=randColor()
    else:
        col=(0,0,0)
    table_3[table_img>0]=col
    table_3=table_3.astype("uint8")
    # table_data
    table_data=table_hw+table_print
    table_data[table_data>0]=255    

    return table_3,table_data,table_cmap,table_wmap
