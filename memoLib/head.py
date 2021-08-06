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
from glob import glob
from .memo import rand_head,Head,LineSection,LineWithExtension
import string
#----------------------------
# print capacity
#----------------------------
def createPrintedMemoHead(ds,language,iden):


    """
        @function author:        
        Create image of top part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
        returns:
            head_img  =  marked memo head image
            labels    =  char labels
            render    =  render properties
            marks     =  available iden to place text    
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

    