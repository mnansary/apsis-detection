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
import os
import cv2

from coreLib.dataset import DataSet
from coreLib.config import config

from coreLib.rendermaps import createNoisyMaps
from coreLib.craft  import gaussian_heatmap
from coreLib.utils import create_dir,LOG_INFO 
from tqdm import tqdm

#--------------------
def set_config(args):
    # number of lines per image
    config.min_num_lines    =int(args.cfg_min_num_lines)
    config.max_num_lines    =int(args.cfg_max_num_lines)
    # number of words per line
    config.min_num_words    =int(args.cfg_min_num_words) 
    config.max_num_words    =int(args.cfg_max_num_words) 
    # number of components in a word if  component type:"grapheme" is used
    config.min_word_len     =int(args.cfg_min_word_len)   
    config.max_word_len     =int(args.cfg_max_word_len)  
    # number of digits  in a number if  component type:"number" is used
    config.min_num_len      =int(args.cfg_min_num_len)  
    config.max_num_len      =int(args.cfg_max_num_len)
    # space between two words/numbers [in pixels]
    config.word_min_space   =int(args.cfg_word_min_space)
    config.word_max_space   =int(args.cfg_word_max_space)
    # space between two lines [in pixels]
    config.vert_min_space   =int(args.cfg_vert_min_space)   
    config.vert_max_space   =int(args.cfg_vert_max_space)
    
    # height dimension for any kind of component
    config.comp_dim         =int(args.cfg_comp_dim)
    # heatmap distance ration if line text format is used
    config.heatmap_ratio    =float(args.cfg_heatmap_ratio)

    # language
    config.data.sources     =[lang for lang in args.cfg_languages]
    # format
    config.data.formats     =[fmt for fmt in args.cfg_formats]
    # components
    config.data.components  =[comp for comp in args.cfg_components]
    


def saveModeData(ds,nb,mode,img_dim):
    '''
        saves data based on format and mode
        args:
            ds      : datset resource
            nb      : number of data to generate
            mode    : save dirs
            img_dim : final data size
    '''
    skipped=[]
    gheatmap=gaussian_heatmap(size=512,distanceRatio=config.heatmap_ratio)

    for i in tqdm(range(nb)):
        try:
            # data execution
            img,hmap,lmap=createNoisyMaps(ds,gheatmap)
            # data formation
            img_path =os.path.join(mode.imgs,f"img{i}.png")
            link_path=os.path.join(mode.linkmaps,f"img{i}.png")
            heat_path=os.path.join(mode.heatmaps,f"img{i}.png")
            # save
            cv2.imwrite(img_path,cv2.resize(img,(img_dim,img_dim)))
            cv2.imwrite(link_path,cv2.resize(lmap,(img_dim,img_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))
            cv2.imwrite(heat_path,cv2.resize(hmap,(img_dim,img_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))
            
        except Exception as e:
            skipped.append(i)

    LOG_INFO(f"Skipped Images:{len(skipped)}")

#--------------------
# main
#--------------------
def main(args):
    # arg setup
    data_dir   =   args.data_dir
    save_dir   =   args.save_dir
    ds_iden    =   args.dataset_iden
    # dimension of the image
    img_dim    =   int(args.cfg_data_dim)
    nb_train   =   int(args.train_samples)
    
    #-------------------------
    # set config
    #------------------------
    set_config(args)
    # log config
    for key,val in vars(config).items():
        if "__" not in key and key!="data":
                LOG_INFO(f"config:{key}={val}")
    LOG_INFO(f"languages:{config.data.sources}")
    LOG_INFO(f"formats:{config.data.formats}")
    LOG_INFO(f"components:{config.data.components}")
    
    #-------------------------
    # resources
    # ------------------------    
    ds=DataSet(data_dir)
    #-------------------------
    # saving
    #------------------------
    
    class save:
        dir=create_dir(save_dir,f"{ds_iden}")
        imgs=create_dir(dir,"images")
        heatmaps=create_dir(dir,"heatmaps")
        linkmaps=create_dir(dir,"linkmaps")

    
    saveModeData(ds,nb_train,save,img_dim)
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Scenetext Detection Dataset Creation Script")
    parser.add_argument("data_dir", help="Path of the base folder under source data folder ")
    parser.add_argument("save_dir", help="Path of the directory to save the dataset")
    parser.add_argument("dataset_iden", help="The desired name for  the dataset.Use something that can help you remember the generation details.Example: (bangla_synth) may indicate only bangla data")
    
    parser.add_argument("--train_samples",required=False,default=10000,help ="number of train samples to create : default=10000")
    
    parser.add_argument("--cfg_data_dim",required=False,default=512,help ="dimension of the image [Since only squre images are produced, providing one value is enough] : default=786")
    parser.add_argument("--cfg_comp_dim",required=False,default=64,help ="height dimension for any kind of component : default=64")
    
    parser.add_argument("--cfg_min_num_lines",required=False,default=1,help ="min number of lines per image : default=1")
    parser.add_argument("--cfg_max_num_lines",required=False,default=10,help ="max number of lines per image : default=10")
    
    parser.add_argument("--cfg_min_num_words",required=False,default=1,help ="min number of words per line : default=1")
    parser.add_argument("--cfg_max_num_words",required=False,default=10,help ="max number of words per line : default=10")
    
    parser.add_argument("--cfg_min_word_len",required=False,default=1,help ="min number of components in a word if  component type:[grapheme] is used : default=1")
    parser.add_argument("--cfg_max_word_len",required=False,default=10,help ="max number of components in a word if  component type:[grapheme] is used : default=10")
    
    parser.add_argument("--cfg_min_num_len",required=False,default=1,help ="min number of digits  in a number if  component type:[number] is used : default=1")
    parser.add_argument("--cfg_max_num_len",required=False,default=10,help ="max number of digits  in a number if  component type:[number] is used : default=10")
    
    parser.add_argument("--cfg_word_min_space",required=False,default=50,help ="min space between two words/numbers [in pixels] : default=50")
    parser.add_argument("--cfg_word_max_space",required=False,default=100,help ="max space between two words/numbers [in pixels] : default=100")
    
    parser.add_argument("--cfg_vert_min_space",required=False,default=1,help ="min space between two lines [in pixels] : default=1")
    parser.add_argument("--cfg_vert_max_space",required=False,default=100,help ="max space between two lines [in pixels] : default=100")
    
    parser.add_argument("--cfg_heatmap_ratio",required=False,default=1.5,help =" heatmap distance ration if line text format is used[float available] : default=2 ")
    
    parser.add_argument('--cfg_languages',nargs='+',required=False,default=["bangla","english"],
                        help='a list of language source to be used|| available:[bangla,english]. e.g., "--cfg_languages bangla englist", or "--cfg_languages bangla" (for single use) ')
    parser.add_argument('--cfg_formats',nargs='+',required=False,default=["handwriten","printed"],
                        help='a list of formats to be used ||available:[handwriten,printed]. e.g., "--cfg_formats handwriten printed"')
    parser.add_argument('--cfg_components',nargs='+',required=False,default=["number","grapheme"],
                        help='a list of type of components to be used ||available:["number","grapheme"]. e.g., "--cfg_components number grapheme"')
    
    
    
    
    
    
    args = parser.parse_args()
    main(args)
    
    