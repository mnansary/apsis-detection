#!/usr/bin/env python
# coding: utf-8
#--------------------------
# imports
#---------------------------
import sys
import argparse
sys.path.append('../')
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
import cv2
import numpy as np  
from coreLib.utils import LOG_INFO,create_dir
from coreLib.craft import gaussian_heatmap,get_maps
#--------------------------
# resources
#---------------------------
def main(args):
    dim=(args.height,args.width)
    img_paths=[img_path for img_path in tqdm(glob(os.path.join(args.data_dir,"*.*")))]
    save_path=create_dir(args.save_path,"icdar")
    img_dir=create_dir(save_path,"images")
    heat_dir=create_dir(save_path,"heatmaps")
    link_dir=create_dir(save_path,"linkmaps")

    gheatmap=gaussian_heatmap(size=1024,distanceRatio=4)

    for img_path in tqdm(img_paths):
        try:
            gt_path=img_path.split(".")[0]
            gt_path=gt_path.replace("Images","GT")
            bmp_path=gt_path+"_GT.bmp"
            txt_path=gt_path+"_GT.txt"
            iden=int(os.path.basename(gt_path))
            # images anon
            img=cv2.imread(img_path)
            map_shape=(img.shape[0],img.shape[1])    
            # maps
            heat_map=np.zeros(map_shape)
            link_map=np.zeros(map_shape)

            # annotations 
            gt_img=cv2.imread(bmp_path)
            gt_img=cv2.cvtColor(gt_img,cv2.COLOR_BGR2RGB)

            with open(txt_path,"r") as tfile:
                lines=tfile.readlines()

            words=[]
            word=[]
            for line in lines:
                if line=="\n":
                    words.append(word)
                    word=[]
                else:
                    word.append(line)
            for word in words:
                num_char=len(word)
                if num_char>1:
                    prev=[[] for _ in range(num_char)]
                else:
                    prev=None
                for idx,data in enumerate(word):
                    data=data.strip().split()
                    if len(data)>3 and data[0][0]!="#" and data[-1].strip():
                        # bbox
                        xmin,ymax,xmax,ymin=data[5:-1]
                        cbbox=[int(xmin),int(ymin),int(xmax),int(ymax)]
                        heat_map,link_map,prev=get_maps(cbbox,
                                                        gheatmap,
                                                        heat_map,
                                                        link_map,
                                                        prev,
                                                        idx)

            # save
            img=cv2.resize(img,dim)
            cv2.imwrite(os.path.join(img_dir,f"{iden}.png"),img)
            heat_map=cv2.resize(heat_map,dim,fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(heat_dir,f"{iden}.png"),heat_map)
            link_map=cv2.resize(link_map,dim,fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(link_dir,f"{iden}.png"),link_map)
        except Exception as e:
            pass


if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("ICDAR to Craft Dataset Creation Script")
    parser.add_argument("data_dir", help="Path to Images folder in icdar folder")
    parser.add_argument("save_path", help="Path to save the processed data")
    parser.add_argument("--height",required=False,default=1024,help ="height dimension of the image : default=1024")
    parser.add_argument("--width",required=False,default=1024,help ="width dimension of the image : default=1024")
    args = parser.parse_args()
    main(args)
    
