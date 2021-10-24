#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def stripPads(arr,val):
  '''
      strip specific values
  '''
  arr=arr[~np.all(arr == val, axis=1)]
  arr=arr[:, ~np.all(arr == val, axis=0)]
  return arr

#---------------------------------------------------------------

def randColor():
    '''
        generates random color
    '''
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#---------------------------------------------------------------------------------------------------------------------
def padDetectionImage(img,gray=False,pad_value=255):
    cfg={}
    if gray:
        h,w=img.shape
    else:
        h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        if gray:
            pad =np.zeros((h,pad_width))
        else:    
            pad =np.ones((h,pad_width,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        if gray:
            pad=np.zeros((pad_height,w))
        else:
            pad =np.ones((pad_height,w,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    if not gray:
        img=img.astype("uint8")
    return img,cfg

'''
    @author: Tahsin Reasat
    Adoptation:MD. Nazmuddoha Ansary
'''

#--------------------
# Parser class
#--------------------
class GraphemeParser():
    def __init__(self):
        self.vds    =['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
        self.cds    =['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
        self.roots  =['ং','ঃ','অ','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ক','ক্ক','ক্ট','ক্ত','ক্ল','ক্ষ','ক্ষ্ণ',
                    'ক্ষ্ম','ক্স','খ','গ','গ্ধ','গ্ন','গ্ব','গ্ম','গ্ল','ঘ','ঘ্ন','ঙ','ঙ্ক','ঙ্ক্ত','ঙ্ক্ষ','ঙ্খ','ঙ্গ','ঙ্ঘ','চ','চ্চ',
                    'চ্ছ','চ্ছ্ব','ছ','জ','জ্জ','জ্জ্ব','জ্ঞ','জ্ব','ঝ','ঞ','ঞ্চ','ঞ্ছ','ঞ্জ','ট','ট্ট','ঠ','ড','ড্ড','ঢ','ণ',
                    'ণ্ট','ণ্ঠ','ণ্ড','ণ্ণ','ত','ত্ত','ত্ত্ব','ত্থ','ত্ন','ত্ব','ত্ম','থ','দ','দ্ঘ','দ্দ','দ্ধ','দ্ব','দ্ভ','দ্ম','ধ',
                    'ধ্ব','ন','ন্জ','ন্ট','ন্ঠ','ন্ড','ন্ত','ন্ত্ব','ন্থ','ন্দ','ন্দ্ব','ন্ধ','ন্ন','ন্ব','ন্ম','ন্স','প','প্ট','প্ত','প্ন',
                    'প্প','প্ল','প্স','ফ','ফ্ট','ফ্ফ','ফ্ল','ব','ব্জ','ব্দ','ব্ধ','ব্ব','ব্ল','ভ','ভ্ল','ম','ম্ন','ম্প','ম্ব','ম্ভ',
                    'ম্ম','ম্ল','য','র','ল','ল্ক','ল্গ','ল্ট','ল্ড','ল্প','ল্ব','ল্ম','ল্ল','শ','শ্চ','শ্ন','শ্ব','শ্ম','শ্ল','ষ',
                    'ষ্ক','ষ্ট','ষ্ঠ','ষ্ণ','ষ্প','ষ্ফ','ষ্ম','স','স্ক','স্ট','স্ত','স্থ','স্ন','স্প','স্ফ','স্ব','স্ম','স্ল','স্স','হ',
                    'হ্ন','হ্ব','হ্ম','হ্ল','ৎ','ড়','ঢ়','য়']

        

    def word2grapheme(self,word):
        graphemes = []
        grapheme = ''
        i = 0
        while i < len(word):
            grapheme += (word[i])
            # print(word[i], grapheme, graphemes)
            # deciding if the grapheme has ended
            if word[i] in ['\u200d', '্']:
                # these denote the grapheme is contnuing
                pass
            elif word[i] == 'ঁ':  
                # 'ঁ' always stays at the end
                graphemes.append(grapheme)
                grapheme = ''
            elif word[i] in list(self.roots) + ['়']:
                # root is generally followed by the diacritics
                # if there are trailing diacritics, don't end it
                if i + 1 == len(word):
                    graphemes.append(grapheme)
                elif word[i + 1] not in ['্', '\u200d', 'ঁ', '়'] + list(self.vds):
                    # if there are no trailing diacritics end it
                    graphemes.append(grapheme)
                    grapheme = ''

            elif word[i] in self.vds:
                # if the current character is a vowel diacritic
                # end it if there's no trailing 'ঁ' + diacritics
                # Note: vowel diacritics are always placed after consonants
                if i + 1 == len(word):
                    graphemes.append(grapheme)
                elif word[i + 1] not in ['ঁ'] + list(self.vds):
                    graphemes.append(grapheme)
                    grapheme = ''

            i = i + 1
            # Note: df_cd's are constructed by df_root + '্'
            # so, df_cd is not used in the code

        return graphemes

    
