#!/usr/bin/env python
# coding: utf-8
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#----------------------
# imports
#----------------------
import sys
sys.path.append('../')
import argparse
import os 
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from coreLib.utils import LOG_INFO,create_dir
from coreLib.craft import gaussian_heatmap,get_maps
from glob import glob
from tqdm.auto import tqdm
tqdm.pandas()
import random
random.seed(42)
#-------------------------------------
# helpers
#-------------------------------------
def extract_info(_dir,coords,fmt):
    '''
        extracts information from boise-state annotations
    '''
    img_paths=[img_path for img_path in glob(os.path.join(_dir,f"*.{fmt}"))]
    liness=[]
    words=[]
    comps=[]
    chars=[]
    xmins=[]
    ymins=[]
    xmaxs=[]
    ymaxs=[]
    _paths=[]
    # images
    for img_path in tqdm(img_paths):
        base=img_path.split(".")[0]
        # text path
        _iden=os.path.basename(img_path).split(".")[0]
        text_path=os.path.join(_dir,coords,f"{_iden}.txt")
        with open(text_path,"r") as tf:
            lines=tf.readlines()
        for line in lines:
            parts=line.split()
            if len(parts)>4:
                line_num=parts[0].replace("\ufeff","")
                word_num=parts[1]
                label=parts[2]
                data=parts[3]
                x,y,w,h=[int(i) for i in parts[-1].split(",")]
                liness.append(line_num)
                words.append(word_num)
                chars.append(label)
                xmins.append(x)
                ymins.append(y)
                xmaxs.append(x+w)
                ymaxs.append(y+h)
                _paths.append(img_path)
                comps.append(data)
    df=pd.DataFrame({"line":liness,
                     "word":words,
                     "char":chars,
                     "comp":comps,
                     "xmin":xmins,
                     "ymin":ymins,
                     "xmax":xmaxs,
                     "ymax":ymaxs,
                     "image":_paths})
    return df

def check_missing(_dir,coords,fmt):
    '''
        checks for missing data
    '''
    img_paths=[img_path for img_path in glob(os.path.join(_dir,f"*.{fmt}"))]
    txt_paths=[txt_path for txt_path in glob(os.path.join(_dir,coords,"*.txt"))]
    # error check
    for img_path in tqdm(img_paths):
        if "jpg" in img_path:
            _iden=os.path.basename(img_path).split(".")[0]
            txt_path=os.path.join(_dir,coords,f"{_iden}.txt")
            if not os.path.exists(txt_path):
                print(img_path)
                for txt in txt_paths:
                    if _iden in txt:
                        print(txt)
                        niden=os.path.basename(txt).split('.')[0]
                        print(f"RENAME:{_iden} to {niden}")
                        os.rename(os.path.join(_dir,f"{_iden}.{fmt}"),
                                  os.path.join(_dir,f"{niden}.{fmt}"))
                        
#-------------------------------------------
# execution
#-------------------------------------------                        
def main(args):

    base_path=os.path.dirname(args.readme_txt_path)
    LOG_INFO(base_path)
    assert len(os.listdir(base_path))==5,"WRONG PATH FOR README.txt"
    dfs=[]
    # ## 1.Camera
    _dir=os.path.join(base_path,'1. Camera','1. Essay')
    coords='Character Coordinates_a'
    fmt="jpg"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))
    # ## 2. Scan
    _dir=os.path.join(base_path,'2. Scan','1. Essay')
    coords='Character Coordinates_a'
    fmt="tif"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))
    ## 3. Conjunct
    _dir=os.path.join(base_path,'3. Conjunct')
    coords='Character Coordinates'
    fmt="tif"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))

    df=pd.concat(dfs,ignore_index=True)


    main_path=create_dir(args.save_path,"bs")
    img_dir =create_dir(main_path,"images")
    hmap_dir=create_dir(main_path,"heatmaps")
    lmap_dir=create_dir(main_path,"linkmaps")


    iden=0
    dim=(args.height,args.width)

    gheatmap=gaussian_heatmap(size=512,distanceRatio=1.5)
    for img_path in tqdm(df.image.unique()):
        idf=df.loc[df.image==img_path]
        #-------------
        # image
        #-------------
        img=cv2.imread(img_path)
        map_shape=(img.shape[0],img.shape[1])
        
        # maps
        heat_map=np.zeros(map_shape)
        link_map=np.zeros(map_shape)
            
        for line in idf.line.unique():
            linedf=idf.loc[idf.line==line]
            for word in linedf.word.unique():
                wdf=linedf.loc[linedf.word==word]
                if len(wdf)>1:
                    prev=[[] for _ in range(len(wdf))]
                else:
                    prev=None
                for idx in range(len(wdf)):
                    #--------------------
                    # heat map
                    #-------------------
                    cxmin=wdf.iloc[idx,4]
                    cymin=wdf.iloc[idx,5]
                    cxmax=wdf.iloc[idx,6]
                    cymax=wdf.iloc[idx,7]
                    heat_map,link_map,prev=get_maps([cxmin,cymin,cxmax,cymax],
                                                    gheatmap,
                                                    heat_map,
                                                    link_map,
                                                    prev,
                                                    idx)
        # save
        img=cv2.resize(img,dim)
        cv2.imwrite(os.path.join(img_dir,f"{iden}.png"),img)
        heat_map=cv2.resize(heat_map,dim,fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(hmap_dir,f"{iden}.png"),heat_map)
        link_map=cv2.resize(link_map,dim,fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(lmap_dir,f"{iden}.png"),link_map)
        iden+=1

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Boise State to Craft Dataset Creation Script")
    parser.add_argument("readme_txt_path", help="Path to The **README.txt** file under **bs**folder")
    parser.add_argument("save_path", help="Path to save the processed data")
    parser.add_argument("--height",required=False,default=1024,help ="height dimension of the image : default=1024")
    parser.add_argument("--width",required=False,default=1024,help ="width dimension of the image : default=1024")
    args = parser.parse_args()
    main(args)
    