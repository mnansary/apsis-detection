from memoLib.dataset import DataSet
from memoLib.utils import create_dir,LOG_INFO
from memoLib.joiner import create_table_data
from tqdm.auto import tqdm
from glob import glob
import os
import cv2
import random

data_dir= "/home/apsisdev/ansary/DATASETS/Detection/source/"
save_dir="/home/apsisdev/ansary/DATASETS/Detection/"
save_dir=create_dir(save_dir,"memo_table")
save_dir=create_dir(save_dir,"styled")
img_dir =create_dir(save_dir,"image")
wmap_dir =create_dir(save_dir,"wordmap")
cmap_dir =create_dir(save_dir,"charmap")
n_data=2000
ds=DataSet(data_dir)
LOG_INFO(save_dir)



dim=(1024,1024)
for i in tqdm(range(n_data)):
    try:
        lang=random.choice(["bangla","english"])
        img,cmap,wmap=create_table_data(ds,lang,random.choice(ds.style_paths))
        img=cv2.resize(img,dim)
        img=cv2.resize(img,(dim[0]//2,dim[1]//2))
        img=cv2.resize(img,dim)
        
        cmap=cv2.resize(cmap,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        wmap=cv2.resize(wmap,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # save
        cv2.imwrite(os.path.join(cmap_dir,f"{i}.png"),cmap)
        cv2.imwrite(os.path.join(wmap_dir,f"{i}.png"),wmap)
        
        # ksize
        ksize = (3, 3)        
        # Using cv2.blur() method 
        img = cv2.blur(img, ksize)
        cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
        
    except Exception as e:
        LOG_INFO(f"Skipping:{i}")
        print(e)