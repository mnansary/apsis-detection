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
import string

from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont

from .memo import rand_head,Head,LineSection,LineWithExtension,Table,rand_products,rand_word
from .word import createPrintedLine,handleExtensions
from .utils import padLineImg,padAllAround,placeWordOnMask
from .table import createTable,tableTextRegions

#----------------------------
# render capacity: toolset
#----------------------------
def renderFontMaps(LineSection,font_path):
    '''
        renders a font map
    '''
    maps={}
    sizes=LineSection.font_sizes_big+LineSection.font_sizes_mid
    for size in sizes:
        maps[str(size)]=PIL.ImageFont.truetype(font_path, size=size)
    return maps

#----------------------------
# render capacity: memo head
#----------------------------
def renderMemoHead(ds,language,iden,font_path=None):


    """
        @function author:        
        Create image of top part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]

    head=Head()
    lineSection=LineSection()
    lineWithExtension=LineWithExtension()
    head=rand_head(graphemes,numbers,head,lineSection,lineWithExtension)
    
    if font_path is None:
        maps=renderFontMaps(lineSection,random.choice(font_paths))
    else:
        maps=renderFontMaps(lineSection,font_path)
    
    ext_sym=random.choice(lineWithExtension.ext_symbols)
    
    h_max=0
    w_max=0
    line_images=[]
    line_labels=[]
    # create line sections
    for line_data in head.line_sections:
        assert len(line_data)==1
        data=line_data[0]
        img,labels,iden=createPrintedLine(iden=iden,words=data["words"],font=maps[str(data["font_size"])],font_size=data["font_size"])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        line_images.append(img)
        line_labels+=labels
    
    line_images=[padLineImg(line_img,h_max,w_max) for line_img in line_images]
    
    # create double ext sections
    for data in head.double_exts:
        assert len(data)==2
        img1,labels1,iden=createPrintedLine(iden=iden,words=data[0]["words"],font=maps[str(data[0]["font_size"])],font_size=data[0]["font_size"])
        # add ext
        h1,w1=img1.shape
        ext_w=w_max//2-w1
        ext=np.ones((h1,ext_w))*iden
        labels1.append({f"{iden}":"ext"})
        iden+=1        
        img1=np.concatenate([img1,ext],axis=1)
        
        img2,labels2,iden=createPrintedLine(iden=iden,words=data[0]["words"],font=maps[str(data[0]["font_size"])],font_size=data[0]["font_size"])
        # add ext
        h2,w2=img2.shape
        ext_w=w_max//2-w2
        ext=np.ones((h2,ext_w))*iden 
        labels2.append({f"{iden}":"ext"})
        iden+=1
        img2=np.concatenate([img2,ext],axis=1)
        
        img=np.concatenate([img1,img2],axis=1)
        # correction
        h,w=img.shape
        if w<w_max:
            pad=np.zeros((h,w_max-w))
            img=np.concatenate([img,pad],axis=1)
        # append
        line_images.append(img)
        line_labels+=labels1+labels2
        

    # create single ext sections
    for line_data in head.single_exts:
        assert len(line_data)==1
        data=line_data[0]
        img,labels,iden=createPrintedLine(iden=iden,words=data["words"],font=maps[str(data["font_size"])],font_size=data["font_size"])
        # add ext
        h,w=img.shape
        ext_w=w_max-w
        ext=np.ones((h,ext_w))*iden
        labels.append({f"{iden}":"ext"})
        iden+=1
        
        img=np.concatenate([img,ext],axis=1)
        # append
        line_images.append(img)
        line_labels+=labels
    
    memo_head=np.concatenate(line_images,axis=0)
    # extention
    exts=[]
    for label in line_labels:
        if "ext" in label.values():
            for k in label.keys():
                exts.append(int(k))
    
    print_mask=np.copy(memo_head)
    #printed mask,labeled mask, image
    head_mask=np.zeros(memo_head.shape)
    for v in exts:
        head_mask[memo_head==v]=v
        print_mask[memo_head==v]=0
        idx=np.where(memo_head==v)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        width=x_max-x_min
        ext_word=handleExtensions(ext_sym,maps[str(32)],width)
        memo_head=placeWordOnMask(ext_word,memo_head,v,memo_head)
        memo_head[memo_head==v]=0

    # img,print,region,labels,iden
    for label in line_labels:
        for k,v in label.items():
            if v!="#":
                memo_head[memo_head==int(k)]=255
            else:
                memo_head[memo_head==int(k)]=0
            
    memo_head=255-memo_head
    return memo_head,print_mask,head_mask,line_labels,iden

#----------------------------
# render capacity: table 
#----------------------------
def renderMemoTable(ds,language,iden):
    """
        @function author:        
        Create image of table part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
    table=Table()
    maps=renderFontMaps(table,random.choice(font_paths))
    line_labels=[]
    
    # fill-up products
    table=rand_products(graphemes,numbers,table)
    ## image
    h_max=0
    w_max=0
    prod_images=[]
    ## create line sections
    for line_data in table.products:
        assert len(line_data)==1
        data=line_data[0]
        img,labels,iden=createPrintedLine(iden=iden,words=data["words"],font=maps[str(data["font_size"])],font_size=data["font_size"])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        prod_images.append(img)
        line_labels+=labels
    
    prod_images=[padLineImg(line_img,h_max,w_max) for line_img in prod_images]
    prod_images=[padAllAround(line_img,table.pad_dim) for line_img in prod_images]
    font_size=data["font_size"]
    
    
    
    # fill headers
    header_images=[]
    
    cell_height=prod_images[0].shape[0]
    w_prod=prod_images[0].shape[1]
    ##serial
    if language=="bangla":
        words=[random.choice(table.serial["bn"])]
    else:
        words=[random.choice(table.serial["en"])]
    img,labels,iden=createPrintedLine(iden,words,font=maps[str(font_size)],font_size=font_size)
    header_images.append(padAllAround(img,table.pad_dim))
    line_labels+=labels
    
    ##column headers
    for i in range(random.randint(table.num_extCOL_min,table.num_extCOL_max)):
        words=[rand_word(graphemes,None,table.word_len_max,table.word_len_min)]
        img,labels,iden=createPrintedLine(iden,words,font=maps[str(font_size)],font_size=font_size)
        if i==0:
            # prod column
            img=padLineImg(img,cell_height,w_prod)
        else:
            img=padLineImg(img,cell_height,img.shape[1]+2*table.pad_dim)
        header_images.append(img)
        line_labels+=labels

    # fill total
    words=[rand_word(graphemes,None,table.word_len_max,table.word_len_min)]
    img,labels,iden=createPrintedLine(iden,words,font=maps[str(font_size)],font_size=font_size)
    total_img=padAllAround(img,table.pad_dim)
    line_labels+=labels
    
    
    # fill serial
    serial_images=[]
    for i in range(len(prod_images)):
        sel_val=str(i)
        word=[v for v in sel_val]
        if language=="bangla":
            word=[numbers[int(v)] for v in word]
        word+=[' ']
        img,labels,iden=createPrintedLine(iden,word,font=maps[str(font_size)],font_size=font_size)
        line_labels+=labels
        serial_images.append(padAllAround(img,table.pad_dim))    




    # table_mask
    table_mask=createTable(len(prod_images)+1,len(header_images)+1,2,[img.shape[1] for img in header_images],cell_height)
    regions,labeled_table=tableTextRegions(table_mask,[img.shape[1] for img in header_images])

    # region fillup 
    printed_mask=np.zeros(table_mask.shape)
    # header regs
    #{"serial":slt_serial, "brand":slt_brand,"total":slt_total,"others":slt_others}
    header_regions=[regions["serial"][0]]+[regions["brand"][0]]+regions["others"]
    for reg_val,word in zip(header_regions,header_images):
        printed_mask=placeWordOnMask(word,labeled_table,reg_val,printed_mask)
        labeled_table[labeled_table==reg_val]=0

    # total fillup
    printed_mask=placeWordOnMask(total_img,labeled_table,regions["total"][0],printed_mask)

    # product fillup
    product_regions=regions["brand"][1:]
    for reg_val,word in zip(product_regions,prod_images):
        printed_mask=placeWordOnMask(word,labeled_table,reg_val,printed_mask)
        labeled_table[labeled_table==reg_val]=0
    
    # serial fillup
    serial_regions=regions["serial"][1:]
    for reg_val,word in zip(serial_regions,serial_images):
        printed_mask=placeWordOnMask(word,labeled_table,reg_val,printed_mask)
        labeled_table[labeled_table==reg_val]=0
    

    table_img=np.copy(printed_mask)
    # img,print,region,labels,iden
    for label in line_labels:
        for k,v in label.items():
            if v!="#":
                table_img[table_img==int(k)]=1
            else:
                table_img[table_img==int(k)]=0
    table_mask[table_mask>0]=1
    table_mask=1-table_mask
    table_img=table_img+table_mask
    table_img[table_img>0]=255
    table_img=255-table_img
    

    # img,print,region,labels,iden 
    return table_img, printed_mask,labeled_table,line_labels,iden
            

