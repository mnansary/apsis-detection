# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
#--------------------
# format
#--------------------

def convertToTotalText(page,labels,rotations=None):
    '''
        @author
        create a function to convert page image to total text format data
        This should not depend on- 
            * language or 
            * type (handwritten/printed) or 
            * data(number/word/symbol)
        args:
            page   :     marked image of a page given at letter by letter 
            labels :     list of markings for each word
        returns:
            whatever is necessary for the total-text format
        FUTURE:
            * Rotation will be added after render class 
    '''
    # your code starts from here 
    # after finalization change returns segment under doc string above
    
    # char mask
    char_mask=np.zeros(page.shape)
    for label in labels:
        for k,v in label.items():
            if v!=' ':
                char_mask[page==k]=255

    char_mask=np.expand_dims(char_mask,axis=-1)
    char_mask=np.concatenate([char_mask,char_mask,char_mask],axis=-1)
    char_mask=char_mask.astype("uint8")
    # word_mask
    word_mask=np.zeros(page.shape)
    for label in labels:
        for k,v in label.items():
            if v!=' ':
                idx = np.where(page==k)
                y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])

                word_mask[y_min:y_max+1,x_min:x_max+1]=255

    word_mask=np.expand_dims(word_mask,axis=-1)
    word_mask=np.concatenate([word_mask,word_mask,word_mask],axis=-1)
    word_mask=word_mask.astype("uint8")
    # bounding box format text file
    text_lines=[]
    
    return char_mask,word_mask,text_lines