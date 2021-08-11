from memoLib.dataset import DataSet
from memoLib.utils import create_dir,LOG_INFO
from memoLib.joiner import create_table_data

data_dir= "/home/apsisdev/ansary/DATASETS/Detection/source/"
save_dir="/home/apsisdev/ansary/DATASETS/Detection/"
save_dir=create_dir(save_dir,"memo_table")
save_dir=create_dir(save_dir,"base")
img_dir =create_dir(save_dir,"images")
tmap_dir =create_dir(save_dir,"data")
wmap_dir =create_dir(save_dir,"hands")
cmap_dir =create_dir(save_dir,"table")
n_data=1000
ds=DataSet(data_dir)



from tqdm.auto import tqdm
import os
import cv2
import random
dim=(256,256)
for i in tqdm(range(n_data)):
    try:
        lang=random.choice(["bangla","english"])
        img,tmap,cmap,wmap=create_table_data(ds,lang)
        img=cv2.resize(img,dim)
        tmap=cv2.resize(tmap,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        cmap=cv2.resize(cmap,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        wmap=cv2.resize(wmap,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # save
        cv2.imwrite(os.path.join(tmap_dir,f"{i}.png"),tmap)
        cv2.imwrite(os.path.join(cmap_dir,f"{i}.png"),cmap)
        cv2.imwrite(os.path.join(wmap,f"{i}.png"),wmap)
        
        # ksize
        ksize = (3, 3)        
        # Using cv2.blur() method 
        img = cv2.blur(img, ksize)
        cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
        
    except Exception as e:
        LOG_INFO(f"Skipping:{i}")
        print(e)