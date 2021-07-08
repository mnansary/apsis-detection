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
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 

from tqdm import tqdm
from glob import glob

from .config import config
from .utils import stripPads
tqdm.pandas()
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
        data=np.zeros(img.shape)
        data[img>0]      =   iden
        imgs.append(data)
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


#-----------------------------------
# wrapper
#----------------------------------
def create_word(iden,
                source_type,
                data_type,
                comp_type,
                ds,
                use_dict=True,
                use_symbols =False):
    '''
        creates a marked word image
        args:
            iden                    :       identifier marking value starting
            source_type             :       bangla/english 
            data_type               :       handwritten/printed                  
            comp_type               :       grapheme/number/mixed
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
            use_symbols             :       wheather to use symbol or not
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
        # using symbols
        #if random.choices(population=[0,1],weights=[0.8, 0.2],k=1)[0]==1:
        #use_symbols =   True
        
        sdf         =   ds.common.symbols.df
        num_sym     =   random.randint(config.min_num_sym,config.max_mun_sym)
        syms=[]
        for _ in range(num_sym):
            idx=random.randint(0,len(sdf)-1)
            syms.append(sdf.iloc[idx,1])
    
        comps+=syms
        df=pd.concat([df,sdf],ignore_index=True)
    
    # process data
    if data_type=="handwritten":
        return createHandwritenWords(iden=iden,df=df,comps=comps)
    else:
        return createPrintedWords(iden=iden,comps=comps,fonts=fonts)



    
        
    