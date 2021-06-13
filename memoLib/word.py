# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
from .config import config
from numpy.core.fromnumeric import compress
import regex 
import numpy as np 
import cv2
import PIL.Image,PIL.ImageDraw,PIL.ImageFont
import random

from .utils import stripPads
from .graphemeParser import GraphemeParser,cleaner
#--------------------
# globals 
#--------------------
gp=GraphemeParser()
bangla_num=["০","১","২","৩","৪","৫","৬","৭","৮","৯"]
#--------------------
# word functions 
#--------------------

def processLine(line):
    '''
        processes line for creating printed text
        args:
            line   :   list of line to process
        returns:
            list of list where each line is divided into grapheme level components 
    '''
    line_data=[]
    # keep track of remaining parts of a line
    remaining_part=line
    # find pure bangla words and numbers
    words=regex.findall(r"[^\x20\x3A-\x40\x5B-\x60\x7B-\x7E-\x7C]+",line)
    # iterate words
    for idx,word in enumerate(words):
        comps=[]
        parts=list(remaining_part.partition(word))
        # find previous parts
        previous_part=parts[0]
        # get remainder parts
        remaining_part="".join(parts[2:])
        
        # check valid string to add component
        if previous_part.strip():
            comps+=[c for c in previous_part]

        # number check
        if any(char in bangla_num for char in word):
            comps+=[g for g in word] 
        # word check
        else:
            _word=""
            for ch in word:
                if ch in cleaner.valid_unicodes:
                    _word+=ch
                else:
                    comps+=gp.word2grapheme(_word)
                    _word=""
                    comps+=[ch]
            
            comps+=gp.word2grapheme(_word)

        # for last word
        if idx==len(words)-1:
            comps+=[c for c in remaining_part]

        comps.append(" ")
        line_data+=comps
        
    return line_data


def createPrintedLine(iden,comps,
                      font_path,
                      font_size):
    '''
        creates printed word image
        args:
            iden    :       identifier marking value starting
            linecomps   :       the list of components
            font_path:      the desired font path 
            font_size:      the size of the font
            
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
    label={}
    font=PIL.ImageFont.truetype(font_path, size=font_size)
    
    # max dim
    min_offset=100
    
    comps=[comp for comp in comps if comp is not None]
    # FIND # 
    if config.ext in comps:
        _exts=comps[comps.index(config.ext):]
        comps=comps[:comps.index(config.ext)]  
    else:
        _exts=None

      
    
    max_dim=len(comps)*font_size+min_offset
    
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    for idx,comp in enumerate(comps):
        if idx < len(comps)-1 and comps[idx+1] in mods:
            comps[idx]+=comps[idx+1]
            comps[idx+1]=None 
            
    comps=[comp for comp in comps if comp is not None]
    # construct labels
    
    imgs=[]
    comp_str=''
    
    for comp in comps:
        if comp==" ":
            comp_str+=config.ext

            
        comp_str+=comp    
        # draw
        image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
        draw = PIL.ImageDraw.Draw(image)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        
        imgs.append(np.array(image))
        if comp==" ":
            comp="space"
        # label
        label[iden] = comp 
        iden+=1
        
        
    # add images
    img=sum(imgs)
    img=stripPads(img,0)
    # offset
    vals=list(np.unique(img))
    vals=sorted(vals,reverse=True)
    vals=vals[:-1]
    # set values
    _img=np.zeros(img.shape)
    for v,l in zip(vals,label.keys()):
        _img[img==v]=l
    # resize
    h,w=_img.shape 
    width= int(font_size* w/h) 
    img=cv2.resize(_img,(width,font_size),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    

    # handle extensions
    if _exts is not None:
        img=handleExtensions(img,len(_exts),iden,font,font_size)
        # label
        label[iden] = "ext"
        iden+=1
    return img,label,iden


def handleExtensions(img,len_ext,iden,font,ext_size):
    '''
        creates/ adds extensions to lines
        arg:
            img     : final staged marked image
            len_ext : length of extension
            iden    : marking end value 
            font    : the image font
            ext_size: proper size of the extensions
    '''
    ext_sym=random.choice(config.date_no.exts)
    # draw
    image = PIL.Image.new(mode='L', size=font.getsize(ext_sym))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=ext_sym, fill=iden, font=font)
    
    ext_img=[np.array(image) for _ in range(len_ext)]
    ext_img=np.concatenate(ext_img,axis=1)
    # reshape
    H,W=img.shape

    h,w=ext_img.shape
    width= int(H* w/h) 
    ext_img=cv2.resize(ext_img,(width,H),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    return np.concatenate([img,ext_img],axis=1)    

