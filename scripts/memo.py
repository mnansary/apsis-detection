# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
from memoLib.dataset import DataSet
from memoLib.utils import create_dir,LOG_INFO
from memoLib.joiner import create_memo_data
from tqdm.auto import tqdm
import os
import cv2
import random


def main(args):
    data_dir=args.data_dir
    save_dir=args.save_dir
    save_dir=create_dir(save_dir,"memo")
    img_dir =create_dir(save_dir,"images")
    wmap_dir =create_dir(save_dir,"linkmaps")
    cmap_dir =create_dir(save_dir,"heatmaps")
    n_data=args.n_data
    ds=DataSet(data_dir)
    LOG_INFO(save_dir)



    for i in tqdm(range(n_data)):
        try:
            lang=random.choice(["bangla","english"])
            img,cmap,wmap=create_memo_data(ds,lang,img_height=args.height)
            # save
            cv2.imwrite(os.path.join(cmap_dir,f"{i}.png"),cmap)
            cv2.imwrite(os.path.join(wmap_dir,f"{i}.png"),wmap)
            cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
        except Exception as e:
            pass


if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic Memo Data Creation Script")
    parser.add_argument("data_dir", help="Path to base data under source")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--height",required=False,default=1024,help ="height dimension of the image : default=1024")
    parser.add_argument("--n_data",required=False,default=1024,help ="number of data to create : default=1024")
    
    args = parser.parse_args()
    main(args)
