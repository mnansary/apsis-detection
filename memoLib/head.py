# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np
import random
import os
import cv2
from glob import glob


# tensorflow 
# input --> output -- func/build--coder
# funct--> good stage
#  

class Head:
    '''
        A memo-head can have the following
        * A random top text like : cash-memo , usually a boxed text
        * company_name
        * distributor_name
        * company_address
        * contact
        * no_data
        * _outlet
        * _route
        * _address
    '''

    _ext  ="[#ext]"
    class company:
        value     =   "এ.স. এন্টারপ্রাইজ"                                           # word  (".","-","/")
        text_size =   128
    class distributor:
        value     =   "ড্রিস্ট্রিবিউটরঃ এম/এস হাশেম ট্রেডিং"                                # word: sym(".","-","/")
        text_size =   96
    class addesss:
        value     =   "হাউস নাম্বারঃ ১২৩৪, বাড্ডা, গুলশান, ঢাকা।"                         # word:,number  sym(".","-","/",",","।")
        text_size =   64
    class contact:
        value     =   "মোবাইল নাম্বারঃ ০১৭২৩৪৫৬৭৮৯"                                   #[word:,number]  
        text_size =   32

    class no_date:
        no          =   "নাম্বারঃ"
        date        =   "তারিখঃ"
        value       =   f"{no}[#ext]{date}[#ext]"
    
    _outlet      =   "আউটলেটের নামঃ[#ext]"
    _route       =   "রাউটঃ[#ext]"
    _address     =   "ঠিকানাঃ[#ext]"




def createMemoHead(ds,Head):


    """
          @function author:
                
                Create image of top part of Memo

            args:
                ds         = dataset object that holds all the paths and resources
                head_names = text values of head part which are unchanged. like 
                              [
                              "এ.স. এন্টারপ্রাইজ",
                              "ড্রিস্ট্রিবিউটরঃ এম/এস হাশেম ট্রেডিং",
                              "হাউস নাম্বারঃ ১২৩৪, বাড্ডা, গুলশান, ঢাকা।",                                <LIST>
                              "মোবাইল নাম্বারঃ ০১৭২৩৪৫৬৭৮৯", 
                              "নাম্বারঃ",
                              "তারিখঃ"
                              ]  

              head_var_names = text values of head part which are included dot dot. like 
                                  [
                                    
                                  "আউটলেটের নামঃ",
                                  "রাউটঃ",                                                       <LIST>
                                  "ঠিকানাঃ"
                                    
                                    ]

              returns:
                    final_img         =  Binary Image after placing text on image. <Image>

                    head_iden_list    =  List of iden numbbers for tracking where (name, route, address) want to insert hand written. <LIST>

                    no_date_iden_list =  List of iden numbbers for tracking where (number and date) want to insert hand written. <LIST>

    """


    ### Add space and dot with last two value of head_names list
    head_names[len(head_names)-1] = head_names[len(head_names)-1]+config.date_no.rep*config.ext ## Date
    head_names[len(head_names)-2] = head_names[len(head_names)-2]+config.date_no.rep*config.ext # No.

    ## Add dot head_var_names list
    for i, p in enumerate(head_var_names):
        dot_len = config.date_no.space-len(p)
        head_var_names[i] = head_var_names[i]+dot_len*config.ext

    ## merge both list (head_names, head_var_names)
    data_Text = head_names + head_var_names
    data = [processLine(line) for line in data_Text]

    # stable-fixed
    iden=3
    imgs=[]
    labels=[]
    h_max,w_max=0,0
    
    fonts=[_font for _font in  glob(os.path.join( ds.bangla.fonts,"*.ttf")) if "ANSI" not in _font]
    font_path=random.choice(fonts)
    # find images and labels
    i = 0
    for line in data: 
        i += 1
        if i==1:
            img,label,iden=createPrintedLine(iden,line,font_path,config.headline1_font_size)
        elif i==2:
            img,label,iden=createPrintedLine(iden,line,font_path,config.headline2_font_size)
        else:
            img,label,iden=createPrintedLine(iden,line,font_path,config.headline3_font_size)
        
        # print(iden)
        h,w=img.shape
        if h>h_max:
            h_max=h
        if w>w_max:
            w_max=w

        imgs.append(img)
        labels.append(label)

    # padding
    #######################################
    h_max+=config.line_pad
    w_max+=config.line_pad
    padded=[]
    for img in imgs:
        img=padImg(img, h_max, w_max)
        padded.append(img)

    ## Merge Padded Images without last 2 values
    img_1=np.concatenate(padded[:len(head_names)-2],axis=0)
    
    ## add space for tracking insertion of handwritten
    no_date_iden_list = [i for i in range(655+len(head_names)-2,len(head_names)+655)]
    iden_n_d = 0
    for i in range(len(head_names)-2, len(head_names)):
        org_img = padded[i]
        h,w = org_img.shape
        dim=(h,w)
        space_img = addSpace(padded[i], no_date_iden_list[iden_n_d])
        resize_space_img=cv2.resize(space_img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        padded[i] = resize_space_img
        iden_n_d+=1

    ## merge last 2 values 
    img_2 = np.concatenate(padded[len(head_names)-2:len(head_names)], axis=1)
    (h_img_2, w_img_2) =  img_2.shape

    ## Need reshape of img_2 to merge img_1 and img_2
    h= img_2.shape[0]
    w= img.shape[1]
    dim = (h, w)
    img_2_resized = cv2.resize(img_2, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

    ## merge img_1 and img_2_resized
    img_3 = np.concatenate([img_1, img_2_resized], axis=0)

    ## add space for tracking insertion of handwritten
    head_iden_list = [i for i in range(555+len(head_names),len(padded)+555)]
    iden_i = 0
    for i in range(len(head_names), len(padded)):
        org_img = padded[i]
        h,w = org_img.shape
        dim=(h,w)
        space_img = addSpace(padded[i], head_iden_list[iden_i])
        resize_space_img=cv2.resize(space_img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        padded[i] = resize_space_img
        iden_i+=1

    ## Merge head_var_names
    im_4 = np.concatenate(padded[len(head_names):], axis=0)

    ## Merge img_3 and img_4: Final imgage
    final_img = np.concatenate([img_3, im_4], axis=0)

    # return
    return final_img, head_iden_list, no_date_iden_list



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