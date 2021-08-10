from memoLib.dataset import DataSet
from memoLib.utils import create_dir,LOG_INFO
from memoLib.joiner import create_memo_data

data_dir= "/home/apsisdev/ansary/DATASETS/Detection/source/"
save_dir="/home/apsisdev/ansary/DATASETS/Detection/"
save_dir=create_dir(save_dir,"memo_segs")
img_dir =create_dir(save_dir,"images")
pr_dir =create_dir(save_dir,"prints")
hw_dir =create_dir(save_dir,"hands")
tb_dir =create_dir(save_dir,"table")
n_data=10
ds=DataSet(data_dir)



from tqdm.auto import tqdm
import os
import cv2
import random
dim=(512,512)
for i in tqdm(range(n_data)):
    try:
        lang=random.choice(["bangla","english"])
        img,pr,hw,tb=create_memo_data(ds,lang)
        img=cv2.resize(img,dim)
        pr=cv2.resize(pr,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        hw=cv2.resize(hw,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        tb=cv2.resize(tb,dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # save
        cv2.imwrite(os.path.join(tb_dir,f"{i}.png"),tb)
        cv2.imwrite(os.path.join(pr_dir,f"{i}.png"),pr)
        cv2.imwrite(os.path.join(hw_dir,f"{i}.png"),hw)
        # ksize
        ksize = (3, 3)        
        # Using cv2.blur() method 
        img = cv2.blur(img, ksize)
        cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
        
    except Exception as e:
        LOG_INFO(f"Skipping:{i}")
        print(e)