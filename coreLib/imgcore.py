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
from tqdm import tqdm

tqdm.pandas()

#--------------------
# helpers
#--------------------
#--------------------------------------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr


#--------------------
# image ops Pure
#--------------------

#--------------------------------------------------------------------------------------------
def createWordImage(img,
                    img_height,
                    img_width):
    '''
        cleans and resizes the image after stripping
        args:
            img         :   numpy array grayscale image
            img_height  :   height for each image
            img_width   :   width for each image
        returns:
            resized clean image
    '''
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # strip
    img=stripPads(arr=img,val=255)
    # resize to char dim
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img

#--------------------
# image ops Synthetic
#--------------------

#--------------------------------------------------------------------------------------------
def getGraphemeImg(img_id,
                   png_dir,
                   img_height):
    '''
        reads and cleans a grapheme image 
        args:
            img_id      :  id of the png file
            png_dir     :  directory that contains the raw images
            img_height  :   height for each grapheme
        returns:
            binary grapheme image
    '''
    img_path  = os.path.join(png_dir,f"{img_id}.png")
    img       = cv2.imread(img_path,0)  
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # strip
    img=stripPads(arr=img,val=255)
    # get shape
    h,w=img.shape
    _w=int(img_height*(w/h))
    # resize to char dim
    img=cv2.resize(img,(_w,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------------------------------------------------------------------------
def getRandomSyntheticData(grapheme_list,
                          label_df,
                          png_dir,
                          img_height,
                          img_width):
    '''
        creates a synthetic data for a given list of graphemes
        args:
            grapheme_list  :  list of graphemes that MUST exist in label_csv
            label_df       :  contains the labeled data from bengalai grapheme dataset  
            png_dir        :  directory that contains the raw images
            img_height     :   height for each grapheme
            img_width      :   width of word images 
            
        returns:
            an image of the lexicon that can be built from the given list
        
        **in the case of a non-found grapheme : 
            returns None for the image and the grapheme
    '''
    imgs=[]
    # iterate over the list
    for grapheme in grapheme_list:
        
        # get corresponding image ids for that grapheme
        grapheme_df=  label_df.loc[label_df.grapheme==grapheme]
        img_ids    =  grapheme_df.image_id.tolist()
        if len(img_ids)>0:
            # select a random one
            img_id     =  random.choice(img_ids)
            # get image
            img        =  getGraphemeImg(img_id=img_id,png_dir=png_dir,img_height=img_height)
            imgs.append(img)
        else:
            return None

    if len(imgs)==0:
        return None
    # corner case
    if len(imgs)==1:
        img=imgs[0]
    else:
        # combine
        img=np.concatenate(imgs,axis=1)
    # resize
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------------------------------------------------------------------------
