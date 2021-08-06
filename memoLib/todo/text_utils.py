# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import cv2 
import numpy as np 
def placeImageOnBackground(img,back,bbox):
    '''
        @author
        places a specific image on a given background at a specific location
        args:
            img   :   greyscale image to place
            back  :   background to place the image
            bbox  :   coordinate of bbox i.e., (y_min,y_max,x_min,x_max)
        location constraint:
            the bounding box centering the (x,y) point can be random
        return:
            back  :   back image after placing 'img'
    '''
    (y_min, y_max, x_min, x_max) = bbox
    ## back: ignore lines of table
    back[y_min:y_max,x_min:x_max]=0
    ## img: resize image
    h_max = abs(y_max-y_min)
    w_max = abs(x_max-x_min)
    dim = (h_max, w_max)
    _img = cv2.resize(img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    ## place "img" on "back"
    back[y_min:y_max,x_min:x_max]=_img
    return back

def drawLine(width, height):
    x1, y1 = 0, height//2
    x2, y2 = width, height//2 
    image = np.zeros((height, width)) * 255
    line_thickness = 2
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness)
    return image



def placeHandTextOnMemoHeadImage(ds, memo_head_img, iden_list_head_var_names, iden_list_no_date):
    
    
    '''
        @author
        places a specific image on a given memo head image at a specific location
        args:
            memo_head_img            :   memo head image to place the image
            iden_list_head_var_names :   list of iden values for finding location of head_var_names (name, route, address)
            iden_list_no_date        :   list of iden values for finding location of number and date
            ds                       :   the dataset object
        location constraint:
            the bounding box centering the (x,y) point can be random
        return:
            memo_head_img            :   placing image given memo head image at a specific location.
    '''
    
    ## greyscale images to place (name, route, date)
    imgs_head_var_names = [create_word(i,"bangla","handwritten","graphemes",ds,use_dict=True)[0] for i in range(1, len(iden_list_head_var_names)+1)]
    for i,img in zip(iden_list_head_var_names,imgs_head_var_names):
        idx = np.where(memo_head_img==i)
        _,w=img.shape
        y_min, y_max, x_min, x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])

        if w<x_min: 
            y_min, y_max, x_min, x_max=y_min, y_max, x_min-w, x_max
        else:
            y_min, y_max, x_min, x_max=y_min, y_max, x_min-w//2, x_max
            
        bbox = (y_min, y_max, x_min, x_max)
            
        memo_head_img=placeImageOnBackground(img,memo_head_img,bbox)
        
    ## greyscale images to place (number, date)
    imgs_num_date = [create_word(i,"bangla","handwritten","number",ds,use_dict=True)[0] for i in range(1, len(iden_list_no_date)+1)]
    for i,img in zip(iden_list_no_date,imgs_num_date):
        idx = np.where(memo_head_img==i)
        _,w=img.shape
        y_min, y_max, x_min, x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        
        if w<x_min: 
            y_min, y_max, x_min, x_max=y_min, y_max, x_min-w, x_max
        else:
            y_min, y_max, x_min, x_max=y_min, y_max, x_min-w//2, x_max
            
        bbox = (y_min, y_max, x_min, x_max)
            
        memo_head_img=placeImageOnBackground(img,memo_head_img,bbox)
            
    return memo_head_img