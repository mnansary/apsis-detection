#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os 
import json
import pandas as pd
from tqdm import tqdm
import sys 

from coreLib.utils import LOG_INFO,create_dir
from coreLib.core  import create_single_data
#--------------------
# main
#--------------------
def main(args):
    '''
    '''  
    data_path   =   args.data_path
    save_path   =   args.save_path
    save_path=create_dir(save_path,args.iden)
    num_data    =   int(args.num_data)
    # ops
    raw_path        = os.path.join(data_path,'RAW')
    raw_nums_path   = os.path.join(data_path,'RAW_NUMS')
    # error check
    LOG_INFO("checking args error")
    if not os.path.exists(raw_path):
        raise ValueError("Wrong Data directory given. No RAW png folder in data path")
    if not os.path.exists(raw_nums_path):
        raise ValueError("Wrong Data directory given. No RAW_NUMS png folder in data path")
    # split
    num_train=int(num_data*0.8)
    num_eval =int(num_data*0.2)              
    LOG_INFO("Create Directory Structre: Train")
    mode_dir=create_dir(save_path,"train")
    img_dir =create_dir(mode_dir,"img")
    df_dir  =create_dir(mode_dir,"df")
    tmap_dir=create_dir(mode_dir,"textmap")
    lmap_dir=create_dir(mode_dir,"linkmap")
    for nimg in tqdm(range(num_train)):
        try:
            img,df,textmap,linkmap=create_single_data(raw_path,raw_nums_path,nimg)
            # save images
            cv2.imwrite(os.path.join(img_dir,f"{nimg}.png"),img)
            cv2.imwrite(os.path.join(tmap_dir,f"{nimg}.png"),textmap)
            cv2.imwrite(os.path.join(lmap_dir,f"{nimg}.png"),linkmap)
            # save df
            df.to_csv(os.path.join(df_dir,f"{nimg}.csv"),index=False)
        except  Exception as e:
            LOG_INFO(f"Error:{nimg}{e}")
            


    LOG_INFO("Create Directory Structre: Test")
    mode_dir=create_dir(save_path,"test")
    img_dir =create_dir(mode_dir,"img")
    df_dir  =create_dir(mode_dir,"df")
    tmap_dir=create_dir(mode_dir,"textmap")
    lmap_dir=create_dir(mode_dir,"linkmap")
    
    for nimg in tqdm(range(num_eval)):
        try:
            img,df,textmap,linkmap=create_single_data(raw_path,raw_nums_path,nimg)
            # save images
            cv2.imwrite(os.path.join(img_dir,f"{nimg}.png"),img)
            cv2.imwrite(os.path.join(tmap_dir,f"{nimg}.png"),textmap)
            cv2.imwrite(os.path.join(lmap_dir,f"{nimg}.png"),linkmap)
            # save df
            df.to_csv(os.path.join(df_dir,f"{nimg}.csv"),index=False)
        except  Exception as e:
            LOG_INFO(f"Error:{nimg}{e}")
               
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Craft Detection Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--num_data",required=False,default=100000,help ="number of image data to create : default=100000")
    parser.add_argument("--iden",required=False,default="baseData",help ="identifier of the created dataset : default=baseData")
    args = parser.parse_args()
    main(args)
    
    