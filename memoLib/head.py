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
from .word import createPrintedLine
import string
#----------------------------
# render capacity
#----------------------------
def renderMemoHead(ds,language,iden):


    """
        @function author:        
        Create image of top part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
        returns:
            render    =  render properties
            iden      =  mark iden end  
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
    render=[]
    # create line sections
    for line_data in head.line_sections:
        img,label,iden=createPrintedLine(iden=iden,comps=line_data["comps"],font_path=random.choice(font_paths),font_size=line_data["font_size"])
        render.append({"img":img,"label":label,"invert":line_data["inverted"]})
    
    # create double ext sections
    for d_ext in head.double_exts:
        font_path=random.choice(font_paths)
        img1,label1,iden=createPrintedLine(iden=iden,comps=d_ext["c1"],font_path=font_path,font_size=d_ext["font_size"],ext=d_ext["ext"],ext_len=d_ext["l1"])
        img2,label2,iden=createPrintedLine(iden=iden,comps=d_ext["c2"],font_path=font_path,font_size=d_ext["font_size"],ext=d_ext["ext"],ext_len=d_ext["l2"])
        label={**label1,**label2}
        img=np.concatenate([img1,img2],axis=1)
        render.append({"img":img,"label":label,"invert":False})

    # create single ext sections
    for s_ext in head.single_exts:
        font_path=random.choice(font_paths)
        img,label,iden=createPrintedLine(iden=iden,comps=s_ext["comps"],font_path=font_path,font_size=d_ext["font_size"],ext=d_ext["ext"],ext_len=d_ext["ext_len"])
        render.append({"img":img,"label":label,"invert":False})
    return render,iden