# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
from numpy.lib.function_base import angle
import pandas as pd
import random
import cv2
import numpy as np
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
from wand.image import Image as WImage
from tqdm import tqdm
from glob import glob

from coreLib.utils import rotate_image

from .config import config
from .utils import stripPads,random_exec
from .craft import get_maps_from_masked_images
tqdm.pandas()

#--------------------
# processing functions 
#--------------------
def get_warped_maps(img,hmap,lmap,warp_vec,coord):
    '''
        returns warped image and new coords
        args:
            img      : image to warp
            hmap     : heat map of the image
            lmap     : link map of the image
            warp_vec : which vector to warp
            coord    : list of current coords
              
    '''
    height,width=img.shape
 
    # construct dict warp
    x1,y1=coord[0]
    x2,y2=coord[1]
    x3,y3=coord[2]
    x4,y4=coord[3]
    # warping calculation
    xwarp=random.randint(0,config.max_warp_perc)/100
    ywarp=random.randint(0,config.max_warp_perc)/100
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(coord),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height),flags=cv2.INTER_NEAREST)
    hmap= cv2.warpPerspective(hmap, M, (width,height),flags=cv2.INTER_NEAREST)
    lmap= cv2.warpPerspective(lmap, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,hmap,lmap,dst

def warp_map_wrapper(img,hmap,lmap):
    '''
    args:
        img      : image to warp
        hmap     : heat map of the image
        lmap     : link map of the image
    '''
    warp_types=["p1","p2","p3","p4"]
    height,width=img.shape

    coord=[[0,0], 
        [width-1,0], 
        [width-1,height-1], 
        [0,height-1]]

    # warp
    for i in range(2):
        if i==0:
            idxs=[0,2]
        else:
            idxs=[1,3]
        if random_exec():    
            idx=random.choice(idxs)
            img,hmap,lmap,coord=get_warped_maps(img,hmap,lmap,warp_types[idx],coord)
    return img,hmap,lmap


def curve_data(img,angle,cangle):
    with WImage.from_array(img) as wimg:
        wimg.virtual_pixel = 'black'
        wimg.distort('arc',(angle,cangle))
        wimg=np.array(wimg)
    return wimg


def curve_maps(img,hmap,lmap):
    '''
    args:
        img      : image to warp
        hmap     : heat map of the image
        lmap     : link map of the image
    '''
    angle=random.randint(30,180)
    cangle=random.choice([0,180])
    img=curve_data(img,angle,cangle)
    hmap=curve_data(hmap,angle,cangle)
    lmap=curve_data(lmap,angle,cangle)
    return img,hmap,lmap


#--------------------
# word functions 
#--------------------

def createHandwritenWords(df,comps,gmap):
    '''
        creates handwriten word image
        args:
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components
            gmap    :       gaussian heatmap
        returns:
            img     :       word image
            hmap    :       heat map of the image
            lmap    :       link map of the image
    '''
    iden=2
    comps=[str(comp) for comp in comps]
    # select a height
    height=config.comp_dim
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]
    # construct labels
    imgs=[]
    for comp in comps:
        c_df=df.loc[df.label==comp]
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=c_df.iloc[idx,2] 
        # read image
        img=cv2.imread(img_path,0)
        # resize
        h,w=img.shape 
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # mark image
        img=255-img
        data=np.zeros(img.shape)
        data[img>0]      =   iden
        imgs.append(data)
        iden+=1

    # maps    
    img=np.concatenate(imgs,axis=1)
    return get_maps_from_masked_images(img,gmap)

def createPrintedWords(gmap,comps,fonts):
    '''
        creates printed word image
        args:
            gmap    :       gaussian heatmap
            comps   :       the list of components
            fonts   :       available font paths 
        returns:
            img     :       word image
            hmap    :       heat map of the image
            lmap    :       link map of the image
    '''
    
    comps=[str(comp) for comp in comps]
    # select a font size
    font_size=config.comp_dim
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
    
    # construct labels
    imgs=[]
    comp_str=''
    for comp in comps:
        comp_str+=comp
        # draw
        image = PIL.Image.new(mode='L', size=font.getsize("".join(comps)))
        draw = PIL.ImageDraw.Draw(image)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        imgs.append(np.array(image))
        
        
    # add images
    img=sum(imgs)
    img=stripPads(img,0)
    # offset
    vals=list(np.unique(img))
    vals=sorted(vals,reverse=True)
    vals=vals[:-1]
    
    _img=np.zeros(img.shape)
    iden=2
    for v in vals:
        _img[img==v]=iden
        iden+=1
    
    # resize
    h,w=_img.shape 
    width= int(font_size* w/h) 
    img=cv2.resize(_img,(width,font_size),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return get_maps_from_masked_images(img,gmap)
    


#-----------------------------------
# wrapper
#----------------------------------
def create_word(gmap,
                word_iden,
                source_type,
                data_type,
                comp_type,
                ds,
                use_dict=True):
    '''
        creates a marked word image
        args:
            gmap                    :       gaussian heatmap
            word_iden               :       word indentifier
            source_type             :       bangla/english 
            data_type               :       handwritten/printed                  
            comp_type               :       grapheme/number/mixed
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
    '''
        
    
    # set resources
    if source_type=="bangla":
        dict_df  =ds.bangla.dictionary 
        
        g_df     =ds.bangla.graphemes.df 
        
        n_df     =ds.bangla.numbers.df 
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path]
    elif source_type=="english":
        dict_df  =ds.english.dictionary 
        
        g_df     =ds.english.graphemes.df 
        
        n_df     =ds.english.numbers.df 
        
        fonts    =[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]

    # component selection 
    if comp_type=="grapheme":
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
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(n_df)-1)
            comps.append(n_df.iloc[idx,1])
        df=n_df
    
    else:
        sdf         =   ds.common.symbols.df
        df=pd.concat([g_df,n_df,sdf],ignore_index=True)
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(df)-1)
            comps.append(df.iloc[idx,1])

    
    # process data
    if data_type=="handwritten":
        img,hmap,lmap=createHandwritenWords(df=df,comps=comps,gmap=gmap)
    else:
        img,hmap,lmap=createPrintedWords(gmap=gmap,comps=comps,fonts=fonts)
    

    # warp
    if random_exec(weights=[0.3,0.7]):
        img,hmap,lmap=warp_map_wrapper(img,hmap,lmap)
    # rotate/curve
    if random_exec(weights=[0.5,0.5]):
        if random_exec(weights=[0.5,0.5]):
            angle=random.randint(-90,90)
            img=rotate_image(img,angle)
            hmap=rotate_image(hmap,angle)
            lmap=rotate_image(lmap,angle)
        else:
            img,hmap,lmap=curve_maps(img,hmap,lmap)

    img[img>0]=word_iden    
    return np.squeeze(img),np.squeeze(hmap),np.squeeze(lmap)


    
        
    