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

#--------------------
# main
#--------------------
def main(args):
    '''
    '''  
    data_path   =   args.data_path
    save_path   =   args.save_path
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
                        
    
               
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--num_data",required=False,default=100000,help ="number of image data to create : default=100000")
    args = parser.parse_args()
    main(args)
    
    