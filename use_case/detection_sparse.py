#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from coreLib.dataset import DataSet
from coreLib.utils import create_dir,LOG_INFO
#data_dir= "/media/ansary/DriveData/Work/bengalAI/datasets/CraftData/source"
#save_dir       = "/media/ansary/DriveData/Work/bengalAI/datasets/CraftData/"

save_dir       = "/home/apsisdev/ansary/DATASETS/DETNEW"
data_dir       = "/home/apsisdev/ansary/DATASETS/synthdata_source"

ds=DataSet(data_dir)


# #create dirs
save_dir   =  create_dir(save_dir,"bangla_sparse")
imgs_dir   =  create_dir(save_dir,"imgs")
charmap_dir  =  create_dir(save_dir,"charmaps")
wordmap_dir  =  create_dir(save_dir,"wordmaps")


# In[ ]:


from coreLib.render import createSceneImage,backgroundGenerator,createImageData
from coreLib.config import config
from coreLib.format import convertToTotalText
backGen=backgroundGenerator(ds,dim=(config.back_dim,config.back_dim))
back=next(backGen)


# In[ ]:


#--------------------
# imports
#--------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
#--------------------
# format
#--------------------

def get_gaussian_heatmap(size=512, distanceRatio=1.5):
    '''
        creates a gaussian heatmap
        This is a fixed operation to create heatmaps
    '''
    # distrivute values
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    # create a value mesh grid
    x, y = np.meshgrid(v, v)
    # spreading heatmap
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')

heatmap=get_gaussian_heatmap(size=1024,distanceRatio=2)


def get_targets(page,labels):
    '''
        @author
        create a function to convert page image to total text format data
        This should not depend on- 
            * language or 
            * type (handwritten/printed) or 
            * data(number/word/symbol)
        args:
            page   :     marked image of a page given at letter by letter 
            labels :     list of markings for each word
        returns:
            whatever is necessary for the total-text format
         
    '''
    # source bbobx of heatmap
    src = np.array([[0, 0], 
                    [heatmap.shape[1], 0], 
                    [heatmap.shape[1], heatmap.shape[0]],
                    [0, heatmap.shape[0]]]).astype('float32')

    # word_mask
    word_mask=np.zeros(page.shape)
    # char mask
    char_mask=np.zeros(page.shape)
    for line_labels in labels:
        _ymins=[]
        _ymaxs=[]
        _xmins=[]
        _xmaxs=[] 

        for label in line_labels:
            for k,v in label.items():
                if v!=' ':
                    char_mask[page==k]=255
                    idx = np.where(page==k)
                    
                    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                    _ymins.append(y_min)
                    _ymaxs.append(y_max)
                    _xmins.append(x_min)
                    _xmaxs.append(x_max)            
            
            x_min=min(_xmins)
            x_max=max(_xmaxs)
            y_min=min(_ymins)
            y_max=max(_ymaxs)
            
            x1=x_min
            y1=y_max
            x2=x_max
            y2=y_max
            x3=x_max
            y3=y_min
            x4=x_min
            y4=y_min
            word_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype('float32') 
            # transforms the bbox and creates the heatmap
            M = cv2.getPerspectiveTransform(src=src,dst=word_points)
            word_mask+= cv2.warpPerspective(heatmap,
                                            M, 
                                            dsize=(word_mask.shape[1],
                                                   word_mask.shape[0])).astype('float32')

    char_mask=char_mask.astype("uint8")
    word_mask=word_mask.astype("uint8")
    return char_mask,word_mask
#     #return char_mask#,word_mask


# In[ ]:


import os
import cv2
from tqdm.auto import tqdm

def saveData(nb):
    '''
        number of images to save
    '''
    for i in tqdm(range(nb)):
        try:
            # data execution
            page,labels=createSceneImage(ds)
            back=createImageData(backGen,page,labels)
            charmap,wordmap=get_targets(page,labels)
            # data formation
            img_path =os.path.join(imgs_dir,f"img{i}.png")
            char_path=os.path.join(charmap_dir,f"img{i}.png")
            word_path=os.path.join(wordmap_dir,f"img{i}.png")
            # save
            cv2.imwrite(img_path,cv2.resize(back,(256,256)))
            cv2.imwrite(char_path,cv2.resize(charmap,(256,256)))
            cv2.imwrite(word_path,cv2.resize(wordmap,(256,256)))
        except Exception as e:
            pass


# In[ ]:


saveData(1500)


# In[ ]:




