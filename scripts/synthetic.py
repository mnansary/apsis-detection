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

from coreLib.render import createSceneImage,backgroundGenerator,createImageData
from coreLib.format import lineText,TotalText
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
    


def saveModeData(ds,backGen,nb,mode,fmt,img_dim):
    '''
        saves data based on format and mode
        args:
            ds      : datset resource
            backGen : background generator
            nb      : number of data to generate
            mode    : train/test
            fmt     : totaltext/linetext
            img_dim : final data size
    '''
    skipped=[]
    if fmt=="linetext":
        gheatmap=gaussian_heatmap(size=512,distanceRatio=config.heatmap_ratio)

    for i in tqdm(range(nb)):
        try:
            # data execution
            page,labels=createSceneImage(ds)
            back=createImageData(backGen,page,labels)
            
            if fmt=="totaltext":
                char_mask,word_mask,text_lines=TotalText(page,labels)
                # data formation
                img_path =os.path.join(mode.imgs,f"img{i}.png")
                char_path=os.path.join(mode.charmaps,f"img{i}.png")
                word_path=os.path.join(mode.wordmaps,f"img{i}.png")
                anno_path=os.path.join(mode.annotations,f"poly_gt_img{i}.txt")
                # save
                cv2.imwrite(img_path,cv2.resize(back,(img_dim,img_dim)))
                cv2.imwrite(char_path,cv2.resize(char_mask,(img_dim,img_dim)))
                cv2.imwrite(word_path,cv2.resize(word_mask,(img_dim,img_dim)))
                with open(anno_path,"w") as f:
                    for line in text_lines:
                        f.write(f"{line}\n")
            elif fmt=="linetext":
                heat_mask,link_mask=lineText(page,labels,gheatmap)    
                # data formation
                img_path =os.path.join(mode.imgs,f"img{i}.png")
                link_path=os.path.join(mode.linkmaps,f"img{i}.png")
                heat_path=os.path.join(mode.heatmaps,f"img{i}.png")
                # save
                cv2.imwrite(img_path,cv2.resize(back,(img_dim,img_dim)))
                cv2.imwrite(link_path,cv2.resize(link_mask,(img_dim,img_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))
                cv2.imwrite(heat_path,cv2.resize(heat_mask,(img_dim,img_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST))
                
        except Exception as e:
            #print(e)
            #LOG_INFO(f"Charecter Size too Short To extract: image number:{i}. Skipping Image",mcolor="red")
            skipped.append(i)
    LOG_INFO(f"Skipped Images:{len(skipped)}")

#--------------------
# main
#--------------------
def main(args):
    # arg setup
    data_dir   =   args.data_dir
    save_dir   =   args.save_dir
    save_fmt   =   args.format
    ds_iden    =   args.dataset_iden

    # dimension of the image
    img_dim    =   int(args.cfg_data_dim)
    
    nb_train   =   int(args.train_samples)
    nb_test    =   int(args.test_samples)
    assert save_fmt in ["totaltext","linetext"],"Wrong output format"

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
    backGen=backgroundGenerator(ds,dim=(config.back_dim,config.back_dim))
    #dummy back
    back=next(backGen)
    #-------------------------
    # saving
    #------------------------
    
    class train:
        dir=create_dir(save_dir,f"{ds_iden}.train")
        imgs=create_dir(dir,"images")
        if save_fmt=="totaltext":
            charmaps=create_dir(dir,"charmaps")
            wordmaps=create_dir(dir,"wordmaps")
            annotations=create_dir(dir,"annotations")
        elif save_fmt=="linetext":
            heatmaps=create_dir(dir,"heatmaps")
            linkmaps=create_dir(dir,"linkmaps")

    class test:
        dir=create_dir(save_dir,f"{ds_iden}.test")
        imgs=create_dir(dir,"images")
        if save_fmt=="totaltext":
            charmaps=create_dir(dir,"charmaps")
            wordmaps=create_dir(dir,"wordmaps")
            annotations=create_dir(dir,"annotations")
        elif save_fmt=="linetext":
            heatmaps=create_dir(dir,"heatmaps")
            linkmaps=create_dir(dir,"linkmaps")

    saveModeData(ds,backGen,nb_train,train,save_fmt,img_dim)
    saveModeData(ds,backGen,nb_test,test,save_fmt,img_dim)
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Scenetext Detection Dataset Creation Script")
    parser.add_argument("data_dir", help="Path of the base folder under source data folder ")
    parser.add_argument("save_dir", help="Path of the directory to save the dataset")
    parser.add_argument("format", help="The desired format for creating the data. Available:totaltext,linetext")
    parser.add_argument("dataset_iden", help="The desired name for  the dataset.Use something that can help you remember the generation details.Example: (bangla_synth) may indicate only bangla data")
    
    parser.add_argument("--train_samples",required=False,default=1500,help ="number of train samples to create : default=1500")
    parser.add_argument("--test_samples",required=False,default=128,help ="number of test samples to create    : default=128")
    
    parser.add_argument("--cfg_data_dim",required=False,default=1024,help ="dimension of the image [Since only squre images are produced, providing one value is enough] : default=1024")
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
    parser.add_argument('--cfg_components',nargs='+',required=False,default=["number","grapheme","mixed"],
                        help='a list of type of components to be used ||available:["number","grapheme","mixed"]. e.g., "--cfg_components number grapheme"')
    
    
    
    
    
    
    args = parser.parse_args()
    main(args)
    
    