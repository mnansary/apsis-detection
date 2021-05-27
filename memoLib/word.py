# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import regex 
import numpy as np 
import cv2
import PIL.Image,PIL.ImageDraw,PIL.ImageFont

from .utils import stripPads
from .graphemeParser import GraphemeParser
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
    words=regex.findall(r"[^\x20-\x2F\x3A-\x40\x5B-\x60\x7B-\x7E-\x7C]+",line)
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
            comps+=gp.word2grapheme(word)
        
        # for last word
        if idx==len(words)-1:
            comps+=[c for c in remaining_part]

        comps.append(" ")
        line_data+=comps

    return line_data


def createPrintedLine(iden,
                      comps,
                      font_path,
                      font_size,
                      space_rep="#"):
    '''
        creates printed word image
        args:
            iden    :       identifier marking value starting
            comps   :       the list of components
            font_path:      the desired font path 
            font_size:      the size of the font
            space_rep:      the replacement of the space charecter
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
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
    font=PIL.ImageFont.truetype(font_path, size=font_size)
    # construct labels
    label={}
    imgs=[]
    comp_str=''
    # add space char
    comps.append(space_rep)

    for comp in comps:
        if comp==" ":
            comp=space_rep
        comp_str+=comp
        
        # draw
        image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
        draw = PIL.ImageDraw.Draw(image)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        
        imgs.append(np.array(image))
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
    _img=cv2.resize(_img,(width,font_size),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return _img,label,iden

