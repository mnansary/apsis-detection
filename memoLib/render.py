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
import matplotlib.pyplot as plt
from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont
from numpy.core.fromnumeric import prod

from .memo import Head,Table,Bottom,LineSection,LineWithExtension
from .memo import rand_head,rand_products,rand_word,rand_bottom

from .word import createPrintedLine,handleExtensions
from .table import createTable,tableTextRegions

from .utils import padToFixedHeightWidth,padAllAround,placeWordOnMask
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
def renderMemoHead(ds,language,max_width):


    """
        @function author:        
        Create image of top part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
        returns:
            binary memo head
            marked print mask
            labeled hw region
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
    #--------------------------------------------
    # text gen section
    #--------------------------------------------
    head=Head()
    lineSection=LineSection()
    lineWithExtension=LineWithExtension()
    head=rand_head(graphemes,numbers,head,lineSection,lineWithExtension)
    maps=renderFontMaps(lineSection,random.choice(font_paths))
    ext_sym=random.choice(lineWithExtension.ext_symbols)
    #--------------------------------------------
    # image gen section
    #--------------------------------------------
    reg_iden=5
    h_max=0
    w_max=max_width
    line_images=[]
    # create line sections
    for line_data in head.line_sections:
        assert len(line_data)==1
        data=line_data[0]
        img=createPrintedLine(line=data["line"],font=maps[str(data["font_size"])])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        line_images.append(img)
        
    line_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in line_images]
    
    # create double ext sections
    for data in head.double_exts:
        LINE1=True
        LINE2=True
        assert len(data)==2
        img1=createPrintedLine(line=data[0]["line"][:-1],font=maps[str(data[0]["font_size"])])
        # add ext
        h1,w1=img1.shape
        ext_w=w_max//2-w1
        if ext_w>0:
            ext=np.ones((h1,ext_w))*reg_iden
            reg_iden+=1        
            img1=np.concatenate([img1,ext],axis=1)
        else:
            LINE1=False
            img1=np.zeros((h1,w_max//2))

        img2=createPrintedLine(line=data[0]["line"],font=maps[str(data[0]["font_size"])])
        # add ext
        h2,w2=img2.shape
        if ext_w>0:    
            ext_w=w_max//2-w2
            ext=np.ones((h2,ext_w))*reg_iden 
            reg_iden+=1
            img2=np.concatenate([img2,ext],axis=1)
        else:
            LINE2=False
            img2=np.zeros((h2,w_max//2))
        
        img=np.concatenate([img1,img2],axis=1)
        # correction
        h,w=img.shape
        if w<w_max:
            pad=np.zeros((h,w_max-w))
            img=np.concatenate([img,pad],axis=1)
        # append
        line_images.append(img)
    
    # create single ext sections
    for line_data in head.single_exts:
        assert len(line_data)==1
        data=line_data[0]
        img=createPrintedLine(line=data["line"],font=maps[str(data["font_size"])])
        # add ext
        h,w=img.shape
        ext_w=w_max-w
        if ext_w>0:
            ext=np.ones((h,ext_w))*reg_iden
            reg_iden+=1
            img=np.concatenate([img,ext],axis=1)
        # append
        line_images.append(img)
    #-----------------------------
    # format masks
    #-----------------------------
    img=np.concatenate(line_images,axis=0)
    printed=np.zeros_like(img)
    region =np.zeros_like(img)
    printed[img==1]=1
    #---------------------------------
    # fix image
    #--------------------------------
    for v in sorted(np.unique(img))[2:]:
        region[img==v]=v
        # ext_image
        idx=np.where(img==v)
        x_min,x_max = np.min(idx[1]), np.max(idx[1])
        width=x_max-x_min
        ext_word=handleExtensions(ext_sym,maps[str(lineSection.font_sizes_mid[-1])],width)
        if ext_word is not None:
            # place
            img=placeWordOnMask(ext_word,img,v,img)
            img[img==v]=0
        
    return img,printed,region

#----------------------------
# render capacity: table 
#----------------------------
def renderMemoTable(ds,language):
    """
        @function author:        
        Create image of table part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
              
        returns:
            binary table head
            marked print mask
            labeled hw region
               
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
    
    #--------------------------------------------
    # product
    #--------------------------------------------
    table=Table()
    maps=renderFontMaps(table,random.choice(font_paths))
    table=rand_products(graphemes,numbers,table)
    # fill-up products
    ## image
    h_max=0
    w_max=0
    prod_images=[]
    prod_cmaps=[]
    prod_wmaps=[]
    ## create line sections
    for line_data in table.products:
        assert len(line_data)==1
        data=line_data[0]
        img,cmap,wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        prod_images.append(img)
        prod_cmaps.append(cmap)
        prod_wmaps.append(wmap)
        
        
    
    prod_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_images]
    prod_images=[padAllAround(line_img,table.pad_dim) for line_img in prod_images]
    
    prod_cmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_cmaps]
    prod_cmaps=[padAllAround(line_img,table.pad_dim) for line_img in prod_cmaps]
    
    prod_wmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_wmaps]
    prod_wmaps=[padAllAround(line_img,table.pad_dim) for line_img in prod_wmaps]
    
    # fixed for all of them now
    font_size   =   data["font_size"]
    cell_height =   prod_images[0].shape[0]
    w_prod      =   prod_images[0].shape[1]
    
    # fill headers
    header_images=[]
    header_wmaps=[]
    header_cmaps=[]
    ##serial
    if language=="bangla":
        word=random.choice(table.serial["bn"])
    else:
        word=random.choice(table.serial["en"])
    img,cmap,wmap=createPrintedLine(word,font=maps[str(font_size)])
    header_images.append(padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim))
    header_cmaps.append(padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim))
    header_wmaps.append(padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim))
    
    ##column headers
    for i in range(random.randint(table.num_extCOL_min,table.num_extCOL_max)):
        word=rand_word(graphemes,None,table.word_len_max,table.word_len_min)
        img,cmap,wmap=createPrintedLine(word[:-1],font=maps[str(font_size)])
        if i==0:
            # prod column
            img=padToFixedHeightWidth(img,cell_height,w_prod)
            cmap=padToFixedHeightWidth(cmap,cell_height,w_prod)
            wmap=padToFixedHeightWidth(wmap,cell_height,w_prod)
        else:
            img=padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim)
            cmap=padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim)
            wmap=padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim)
        
        header_images.append(img)
        header_cmaps.append(cmap)
        header_wmaps.append(wmap)
        
        

    # fill total
    word=rand_word(graphemes,None,table.word_len_max,table.word_len_min)
    img,cmap,wmap=createPrintedLine(word[:-1],font=maps[str(font_size)])
    
    total_img=padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim)
    total_cmap=padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim)
    total_wmap=padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim)
    
    # fill serial
    serial_images=[]
    serial_cmaps=[]
    serial_wmaps=[]
    
    serial_width=header_images[0].shape[1]
    for i in range(len(prod_images)):
        sel_val=str(i+1)
        word="".join([v for v in sel_val])
        if language=="bangla":
            word="".join([numbers[int(v)] for v in word])
        
        img,cmap,wmap=createPrintedLine(word,font=maps[str(font_size)])
        serial_images.append(padToFixedHeightWidth(img,cell_height,serial_width))    
        serial_cmaps.append(padToFixedHeightWidth(cmap,cell_height,serial_width))    
        serial_wmaps.append(padToFixedHeightWidth(wmap,cell_height,serial_width))    
    



    # table_mask
    table_mask=createTable(len(prod_images)+1,len(header_images)+1,2,[img.shape[1] for img in header_images],cell_height)
    regions,region=tableTextRegions(table_mask,[img.shape[1] for img in header_images])
    # region fillup 
    printed=np.zeros(table_mask.shape)
    printed_cmap=np.zeros(table_mask.shape)
    printed_wmap=np.zeros(table_mask.shape)
    # header regs
    #{"serial":slt_serial, "brand":slt_brand,"total":slt_total,"others":slt_others}
    header_regions=[regions["serial"][0]]+[regions["brand"][0]]+regions["others"]
    for reg_val,word,cmap,wmap in zip(header_regions,header_images,header_cmaps,header_wmaps):
        printed=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        region[region==reg_val]=0

    # total fillup
    printed=placeWordOnMask(total_img,region,regions["total"][0],printed,fill=True)
    printed_cmap=placeWordOnMask(total_cmap,region,regions["total"][0],printed_cmap,fill=True)
    printed_wmap=placeWordOnMask(total_wmap,region,regions["total"][0],printed_wmap,fill=True)
    
    region[region==regions["total"][0]]=0
    # product fillup
    product_regions=regions["brand"][1:]
    for reg_val,word,cmap,wmap in zip(product_regions,prod_images,prod_cmaps,prod_wmaps):
        printed=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        
        region[region==reg_val]=0
    
    # serial fillup
    serial_regions=regions["serial"][1:]
    for reg_val,word,cmap,wmap in zip(serial_regions,serial_images,serial_cmaps,serial_wmaps):
        printed=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        
        region[region==reg_val]=0
    
    img=np.copy(printed)
    table_mask[table_mask>0]=1
    table_mask=1-table_mask
    img=img+table_mask
    img[img>0]=1
    
    return img,printed,region,printed_cmap,printed_wmap
            
#----------------------------
# render capacity: bottom 
#----------------------------
def renderMemoBottom(ds,language,max_width,pad_dim=10):
    """
        @function author:        
        Create image of table part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
        returns:
            binary img
            marked print mask
            labeled hw region
        
    """
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
    bottom=Bottom()
    maps=renderFontMaps(bottom,random.choice(font_paths))
    # fill-up texts
    bottom=rand_bottom(graphemes,numbers,bottom)
    ## image
    h_max=0
    w_max=0
    
    
    ## create line sections
    data=bottom.sender_line[0]
    sender_img=createPrintedLine(line=data["line"],font=maps[str(data["font_size"])])
    h,w=sender_img.shape
    if h>h_max:h_max=h
    if w>w_max:w_max=w
    
    data=bottom.reciver_line[0]
    
    rec_img=createPrintedLine(line=data["line"],font=maps[str(data["font_size"])])
    h,w=rec_img.shape
    if h>h_max:h_max=h
    if w>w_max:w_max=w
    
    sign_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in [sender_img,rec_img]]


    data=bottom.middle_line[0]
    mid_img=createPrintedLine(line=data["line"],font=maps[str(data["font_size"])])
    mid_img=padToFixedHeightWidth(mid_img,mid_img.shape[0]+2*pad_dim,max_width)
    
    
    mid_pad=np.zeros((h_max,max_width//2))

    sign_images=[sign_images[0]*3,mid_pad,sign_images[-1]*4]
    sign_img=np.concatenate(sign_images,axis=1)

    h,w=sign_img.shape
    sign_img=padToFixedHeightWidth(sign_img,sign_img.shape[0]+2*pad_dim,max_width)
    # print_mask
    if random.choice([0,1])==1:
        printed=np.concatenate([sign_img,mid_img],axis=0)
    else:
        printed=np.concatenate([mid_img,sign_img],axis=0)
    # image mask
    img=np.copy(printed)
    region=np.zeros_like(img)
    rid=5
    #######################
    # fixed region
    for i in [3,4]:
        idx=np.where(img==i)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])            
        region[y_min:y_max,x_min:x_max]=rid
        rid+=1
    #######################
    img[img>0]=1
    printed[img>0]=1

    return img,printed,region