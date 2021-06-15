# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------

from .text_utils import placeImageOnBackground, drawLine
from .utils import padImg
from .word import createPrintedLine, processLine, addSpace, create_word
from .config import config
import random
import os
from glob import glob
import cv2
import numpy as np

def memoBottomFunc(ds,bottom_names_snd_rcv, bottom_names_mdl):

    """
          @function author:
                
                Create image of bottom part of Memo

            args:
                ds                   = dataset object that holds all the paths and resources

                bottom_names_snd_rcv = text values of bottom (sender, reciver). like 
                                      [
                                        "প্রেরকের সাক্ষর", 
                                        "প্রাপকের সাক্ষর"                             <LIST>

                                      ]  

                bottom_names_mdl     = text values of bottom middle part . like 
                                     [
                                       "ব্রি. দ্র. এই খানে কিছু লেখা।"                    <LIST>
                                    
                                    ]

              returns:
                    final_img         =  Binary Image after placing text on image.
                    sr_iden_list      =  iden value of sender and reciever        <LIST>

    """

    ## merge both list (bottom_names_snd_rcv bottom_names_mdl )
    data_Text =  bottom_names_snd_rcv + bottom_names_mdl

    ## Create Function: for Process the Text data
    data = [processLine(line) for line in data_Text]
    
    ## choose fond
    fonts=[_font for _font in  glob(os.path.join( ds.bangla.fonts,"*.ttf")) if "ANSI" not in _font]
    font_path=random.choice(fonts)

    #### sender and reciever text image
    # stable-fixed
    iden=3
    h_max,w_max=0,0
    # find images and labels
    imgs_sr=[]
    labels_sr=[]
    sr_iden_list = [i for i in range(855,len(bottom_names_snd_rcv)+855)]
    iden_sr = 0
    for line in data[:-len(bottom_names_mdl)]: 
        img,label,iden=createPrintedLine(iden,line,font_path,config.headline3_font_size)
        h,w=img.shape
        img_line = drawLine(w, h//2)
        img_line_ = addSpace(img_line, sr_iden_list[iden_sr])
        img_line__=cv2.resize(img_line_, (w, h//2), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        iden_sr += 1
        ## merge image line and sender or reciever image
        _img = np.concatenate([img_line__, img], axis=0)

        h,w=_img.shape
        if h>h_max:
            h_max=h
        if w>w_max:
            w_max=w

        imgs_sr.append(_img)
        labels_sr.append(label)

    h_max+=config.line_pad
    w_max+=config.line_pad
    padded_sr=[]
    for img in imgs_sr:
        # print(img.shape, h_max, w_max)
        img=padImg(img, h_max, w_max)
        padded_sr.append(img)
    
    ### track for resizing middle image
    re_h_max = h_max
    #### middle text image
    h_max,w_max=0,0
    imgs_mdl = []
    labels_mdl = []
    data_mdl = data[len(bottom_names_snd_rcv):]
    for line in data_mdl: 
        img,label,iden=createPrintedLine(iden,line, font_path, config.headline3_font_size)
        h,w=img.shape
        if h>h_max:
            h_max=h
        if w>w_max:
            w_max=w

        imgs_mdl.append(img)
        labels_mdl.append(label)

    h_max+=config.line_pad
    w_max+=config.line_pad
    padded_mdl=[]
    for img in imgs_mdl:
        # print(img.shape, h_max, w_max)
        img=padImg(img, h_max, w_max)
        _,w=img.shape 
        img=cv2.resize(img, (w, re_h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # print(img.shape, h_max, w_max)
        padded_mdl.append(img)
    
    ## Merge Padded Images 
    img_merge=np.concatenate([padded_sr[0], padded_mdl[len(padded_mdl)-1], padded_sr[len(padded_sr)-1]],axis=1)
    
    ## add pad (extend image size)
    h,w = img_merge.shape
    img_merge_pad = padImg(img_merge, h*2, w)
    
    return img_merge_pad, sr_iden_list




def placeSignsOnMemoBottomImage(ds, final_img, sr_iden_list):

    """
          @function author:
                
                places a specific image on a given memo bottom image at a specific location

            args:
                ds                   = dataset object that holds all the paths and resources

                final_img         =  Binary Image after placing text on image.

                sr_iden_list      =  iden value of sender and reciever        <LIST>

              returns:
                    final_img         =  Binary Image after placing singnature text on image.
                   

    """
    ## load sign immages 
    imgss = []
    for i in range(20):
        k = random.randint(0, 3238)
        img_path=os.path.join(ds.common.noise.sign,"mixed_" + str(k) +".bmp")
        if os.path.exists(img_path):
            img=cv2.imread(img_path,0)
            imgss.append(img)

    num_to_select = len(sr_iden_list)          # set the number to select here.
    list_of_random_sign_imgs = random.sample(imgss, num_to_select)

    i_check = 0
    for i,img in zip(sr_iden_list,list_of_random_sign_imgs):
        idx = np.where(final_img==i)
        y_min, y_max, x_min, x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        _, w = final_img.shape

        y_min = 5
        y_max = 165
        if i_check == 0:
            x_min = 10
            x_max = x_max
            
        if i_check == 1:
            x_min = w-w//4
            x_max = x_max

        bbox = (y_min, y_max, x_min, x_max)

        final_img=placeImageOnBackground(img,final_img,bbox)
        
        i_check += 1

    return final_img
