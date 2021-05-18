# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
import math
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 

from tqdm import tqdm
from glob import glob

from .config import config
from .utils import stripPads
tqdm.pandas()
#--------------------
# word functions 
#--------------------
def createHandwritenWords(iden,
                         df,
                         comps,
                         img_dir):
    '''
        creates handwriten word image
        args:
            iden    :       identifier marking value starting
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components
            img_dir :       the directory that contains images 
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
    comps=[str(comp) for comp in comps]
    # select a height
    height=config.comp_dim
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]
    # construct labels
    label={}
    imgs=[]
    for comp in comps:
        c_df=df.loc[df.label==comp]
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=os.path.join(img_dir,f"{c_df.iloc[idx,0]}.bmp") 
        # read image
        img=cv2.imread(img_path,0)
        # resize
        h,w=img.shape 
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # mark image
        img=255-img
        img[img>0]      =   iden
        imgs.append(img)
        # label
        label[iden] = comp 
        iden+=1
    img=np.concatenate(imgs,axis=1)
    return img,label,iden

def createPrintedWords(iden,
                       comps,
                       fonts):
    '''
        creates printed word image
        args:
            iden    :       identifier marking value starting
            comps   :       the list of components
            fonts   :       available font paths 
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
    # sets the value
    val_offset=iden

    comps=[str(comp) for comp in comps]
    # select a font size
    font_size=config.comp_dim
    # max dim
    min_offset=100
    max_dim=len(comps)*font_size+min_offset
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    for idx,comp in enumerate(comps):
        if idx < len(comps)-1 and comps[idx+1] in mods:
            comps[idx]+=comps[idx+1]
            comps[idx+1]=None 
            
    comps=[comp for comp in comps if comp is not None]
    # font path
    font_path=random.choice(fonts)
    font=PIL.ImageFont.truetype(font_path, size=font_size)
    # sizes of comps
    # comp_sizes = [font.font.getsize(comp) for comp in comps] 
    # construct labels
    label={}
    imgs=[]
    x=0
    y=0
    comp_str=''
    for comp in comps:
        comp_str+=comp
        # # calculate increment
        # (comp_width,_),(offset,_)=comp_size
        # dx = comp_width+offset 
        # draw
        image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
        draw = PIL.ImageDraw.Draw(image)
        #draw.text(xy=(x, y), text=comp, fill=iden, font=font)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        
        imgs.append(np.array(image))
        # x+=dx
        # label
        label[iden] = comp 
        iden+=1
        
        
    # add images
    img=sum(imgs)
    img=stripPads(img,0)
    # offset
    img[img>0]+=val_offset-1
    return img,label,iden


#-----------------------------------
# wrapper
#----------------------------------
def create_word(iden,
                source_type,
                data_type,
                comp_type,
                ds,
                use_dict=True):
    '''
        creates a marked word image
        args:
            iden                    :       identifier marking value starting
            source_type             :       bangla/english 
            data_type               :       handwritten/printed                  
            comp_type               :       grapheme/number
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
    '''
    # set resources
    if source_type=="bangla":
        dict_df  =ds.bangla.dictionary 
        
        g_df     =ds.bangla.graphemes.df 
        g_dir    =ds.bangla.graphemes.dir
        
        n_df     =ds.bangla.numbers.df 
        n_dir    =ds.bangla.numbers.dir
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path]
    elif source_type=="english":
        dict_df  =ds.english.dictionary 
        
        g_df     =ds.english.graphemes.df 
        g_dir    =ds.english.graphemes.dir
        
        n_df     =ds.english.numbers.df 
        n_dir    =ds.english.numbers.dir
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]

    # component selection 
    if comp_type=="grapheme":
        img_dir=g_dir
        # dictionary
        if use_dict:
            # select index from the dict
            idx=random.randint(0,len(dict_df)-1)
            comps=dict_df.iloc[idx,1]
        else:
            # construct random word with grapheme
            comps=[]
            len_word=random.randint(config.min_word_len,config.max_word_len)
            for _ in range(len_word):
                idx=random.randint(0,len(g_df)-1)
                comps.append(g_df.iloc[idx,1])
        df=g_df
    
    elif comp_type=="number":
        img_dir=n_dir
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(n_df)-1)
            comps.append(n_df.iloc[idx,1])
        df=n_df

    # process data
    if data_type=="handwritten":
        return createHandwritenWords(iden=iden,df=df,comps=comps,img_dir=img_dir)
    else:
        return createPrintedWords(iden=iden,comps=comps,fonts=fonts)



    
        
    