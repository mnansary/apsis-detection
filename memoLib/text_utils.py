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