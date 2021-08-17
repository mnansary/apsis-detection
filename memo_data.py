from memoLib.dataset import DataSet
from memoLib.utils import create_dir,LOG_INFO
from memoLib.joiner import create_memo_data
from tqdm.auto import tqdm
import os
import cv2
import random

data_dir= "/home/apsisdev/ansary/DATASETS/Detection/source/"
save_dir="/home/apsisdev/ansary/DATASETS/Detection/"
save_dir=create_dir(save_dir,"memo")
img_dir =create_dir(save_dir,"image")
wmap_dir =create_dir(save_dir,"wordmap")
cmap_dir =create_dir(save_dir,"charmap")
n_data=2000
ds=DataSet(data_dir)
LOG_INFO(save_dir)



for i in tqdm(range(n_data)):
    try:
        lang=random.choice(["bangla","english"])
        img,cmap,wmap=create_memo_data(ds,lang)
        # save
        cv2.imwrite(os.path.join(cmap_dir,f"{i}.png"),cmap)
        cv2.imwrite(os.path.join(wmap_dir,f"{i}.png"),wmap)
        cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
    except Exception as e:
        pass