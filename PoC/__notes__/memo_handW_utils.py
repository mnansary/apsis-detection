import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
import sys
import numpy
import random
import scipy.ndimage as sni
from pprint import pprint

from __notes__.memo_utils import drawsPrintTextOnTable
from coreLib.word import create_word

def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat,rotation_mat

def placeHandwrittenTextOnTableImg(printTextData,
                                   total, 
                                   class_map_csv_path, 
                                   font_path,
                                   ds,
                                   iden=2
                                    ):
    
    """
          @function author:
          
          Place Print Bangla text and handwritten text on specific loactions of Binary Image.

          args:
              printTextData   =    format of example dictionary: dict_ = {'serial': ["সেরিঃ",  
                                                                                    "১",
                                                                                    "২",
                                                                                    "৩", 
                                                                                    "৪", 
                                                                                    "৫" ],
                                                                        'brand': ["ব্র্যান্ডের নাম", 
                                                                                    "র‍্যান্ডম প্রোডাক্ট নাম্বার ১",      : type <DICT>
                                                                                    "র‍্যান্ডম বিড়ি",
                                                                                    "আরো কিছু",
                                                                                    "৪২০ মার্কা বিড়ি",
                                                                                    "রং নাম্বার"],
                                                                        'quantity': ["কোয়ান্টিটি"],
                                                                        'rate': ["রেইট/গ্রোস"], 
                                                                        'taka': ["টাকা"]
                                                                        
                                                                      } 
                                    keys of the dictioary mention columns's name ("সেরিঃ","ব্র্যান্ডের নাম", "কোয়ান্টিটি", "রেইট/গ্রোস", "টাকা").
                                    Others are serials and products name. 

              total               =  text of total box (N.B., you can insert no value (None)). (example:total = ["টোটাল"])  : type <LIST>  

              class_map_csv_path  =  path of "classes.csv"  <'./class_map.csv'>    
              
              font_path           =  the desired font path. <'/Bangla.ttf'> 
              
              ds                  =  DataSet for Symbol (numbers hand written, bangla hand written)
              
          returns:
              TableImg            =  Binary Image after placing pritten text and hand written on desired locations.

    """
    
    
    ## call func "drawsPrintTextOnTable()" for creating table image with text 
    Table_Image_with_Text, all_locs, labeled_img = drawsPrintTextOnTable(printTextData, total, class_map_csv_path, font_path)
    
    ## find locations where you want to place handwritten text
    others_locs = [i for i in range(1, all_locs[-1]+2) if i not in all_locs if i != all_locs[len(all_locs)-1]-1]
    
    ## number of columns
    num_cols = len(printTextData)
    
    ## select some locations where you want to place handwritten text
    List = [i for i in range(1,7)]
    N = random.choice(List) # single item from the List
    Updated_others_locs = random.sample(others_locs, N)
    
#     print(others_locs)
#     print(others_locs[len(others_locs)-1])
#     print(Updated_others_locs)
#     print(len(Updated_others_locs))
    
    ## Check whether you want to place rotate text or straight text
    rotated_check=[bool(i) for i in np.array(np.random.randint(0,2,len(Updated_others_locs)))] # For TRUE
    rot=0

    imgs = []
    for i in range(2, len(Updated_others_locs)+2):
        img,label,iden=create_word(i,
                                    "bangla",
                                    "handwritten",
                                    "number",
                                    ds=ds,
                                    use_dict=True)


        if rotated_check[rot] == True:
            List_degree = [i for i in range(10,45+5,5)]
            N_deg = random.choice(List_degree) # single item from the List
            rotated_img,M=rotate_image(img,N_deg)
            imgs.append(rotated_img)

        else:
            imgs.append(img)

        rot += 1
        
#     print(len(imgs))
    
    for i,img in zip(Updated_others_locs,imgs):
        idx = np.where(labeled_img==i)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])

        ## resize bbox
        List_extend_value = [i for i in range(30,60,5)]
        N_extend = random.choice(List_extend_value) # single item from the List
        if i % num_cols == 0 or i == all_locs[len(all_locs)-1]-1 or i==others_locs[len(others_locs)-1]:
            y_min,y_max,x_min,x_max = y_min-N_extend,y_max+N_extend,x_min,x_max
        else:
            y_min,y_max,x_min,x_max = y_min-N_extend,y_max+N_extend,x_min-N_extend,x_max+N_extend

        ## ignore lines of table
        Table_Image_with_Text[y_min:y_max,x_min:x_max]=0

        ## resize image
        h_max = abs(y_max-y_min)
        w_max = abs(x_max-x_min)
        dim = (h_max, w_max)
        _img = cv2.resize(img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

        ## place hand text into text image
        Table_Image_with_Text[y_min:y_max,x_min:x_max]=_img

    return Table_Image_with_Text
        
    
