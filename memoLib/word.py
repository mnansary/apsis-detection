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
import os
from glob import glob 
import PIL.Image,PIL.ImageDraw,PIL.ImageFont
import random
import pandas as pd 

from .config import config
from .utils import stripPads, padImg
from .graphemeParser import GraphemeParser,cleaner
#--------------------
# globals 
#--------------------
gp=GraphemeParser()
bangla_num=["০","১","২","৩","৪","৫","৬","৭","৮","৯"]
#--------------------
# word functions 
#--------------------
def addSpace(img,iden):
    '''
        adds a space at the end of the word
    '''
    h,_=img.shape
    width=random.randint(config.word_min_space,config.word_max_space)
    space=np.ones((h,width))*iden
    return np.concatenate([img,space],axis=1)

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





def createHandwritenWords(iden,
                         df,
                         comps):
    '''
        creates handwriten word image
        args:
            iden    :       identifier marking value starting
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components 
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
        img_path=c_df.iloc[idx,2] 
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
    # add space
    img=addSpace(img,iden)
    label[iden]=' '
    iden+=1
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
    vals=list(np.unique(img))
    vals=sorted(vals,reverse=True)
    vals=vals[:-1]
    
    _img=np.zeros(img.shape)
    for v,l in zip(vals,label.keys()):
        _img[img==v]=l
    
    # add space
    _img=addSpace(_img,iden)
    label[iden]=' '
    iden+=1
    # resize
    # resize
    h,w=_img.shape 
    width= int(font_size* w/h) 
    
    _img=cv2.resize(_img,(width,font_size),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return _img,label,iden


## Create Function: find images and labels
def imgsLabels(ds, processTextData):
    imgs=[]
    labels=[]
    h_max,w_max=0,0
    iden=3

    ## choose fond
    fonts=[_font for _font in  glob(os.path.join( ds.bangla.fonts,"*.ttf")) if "ANSI" not in _font]
    font_path=random.choice(fonts)

    # find images and labels
    for line in processTextData:                     
        img,label,iden=createPrintedLine(iden,line,font_path,config.headline3_font_size)
        h,w=img.shape
        if h>h_max:
            h_max=h
        if w>w_max:
            w_max=w

        imgs.append(img)
        labels.append(label)

    h_max+=config.line_pad
    w_max+=config.line_pad
    padded=[]
    for img in imgs:
        img=padImg(img,config.height_box,w_max) ### <<<<<================= Call Function
        padded.append(img)

    return imgs, labels, padded, w_max

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
            comp_type               :       grapheme/number/mixed
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
    '''
    # using symbols
    if random.choices(population=[0,1],weights=[0.8, 0.2],k=1)[0]==1:
        use_symbols =   True
        sdf         =   ds.common.symbols.df
        num_sym     =   random.randint(config.min_num_sym,config.max_mun_sym)
        syms=[]
        for _ in range(num_sym):
            idx=random.randint(0,len(sdf)-1)
            syms.append(sdf.iloc[idx,1])
    else:
        use_symbols =   False
        
    
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
        use_symbols=False
        sdf         =   ds.common.symbols.df
        df=pd.concat([g_df,n_df,sdf],ignore_index=True)
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(df)-1)
            comps.append(df.iloc[idx,1])

    # moderation for using symbols
    if use_symbols:
        comps+=syms
        df=pd.concat([df,sdf],ignore_index=True)
    
    # process data
    if data_type=="handwritten":
        return createHandwritenWords(iden=iden,df=df,comps=comps)
    else:
        return createPrintedWords(iden=iden,comps=comps,fonts=fonts)



    
        
    
