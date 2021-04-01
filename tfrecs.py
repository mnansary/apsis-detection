#!/usr/bin/python3
# -*-coding: utf-8 -
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import random
import argparse

from glob import glob
from tqdm import tqdm

from coreLib.utils import LOG_INFO,create_dir
from coreLib.store import Processor
#---------------------------------------------------------------
#---------------------------------------------------------------

def main(args):
    '''
        preprocesses data for training
        args:
            data_path   =   the location of folder that contains test and train 
            save_path   =   path to save the tfrecords
            data_size   =   the size of tfrecords
            
    '''
    data_path   =   args.data_path
    save_path   =   args.save_path
    data_size   =   int(args.data_size)
    processor_obj=Processor(data_path,save_path,data_size)
    processor_obj.process()

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("script to create tfrecords")
    parser.add_argument("data_path", help="Path of the data folder that contains Test and Train")
    parser.add_argument("save_path", help="Path to save the tfrecords")
    parser.add_argument("--data_size",required=False,default=500,help ="the size of tfrecords")
    args = parser.parse_args()
    main(args)
    