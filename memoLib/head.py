# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np 
from .word import createPrintedLine, processLine
from .config import config
from .utils import padImg
import random

def memoHeadFunc(ds,head_names,head_var_names):


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
                    final_img            =  Binary Image after placing text on image.

    """


    ### Add space and dot with last two value of head_names list
    head_names[len(head_names)-1] = head_names[len(head_names)-1]+config.date_no.rep*config.ext ## Date
    head_names[len(head_names)-2] = head_names[len(head_names)-2]+config.date_no.rep*config.ext # No.

    ## Add dot head_var_names list
    for i, p in enumerate(head_var_names):
        dot_len = config.date_no.space-len(p)
        head_var_names[i] = head_var_names[i]+dot_len*config.ext
        
    # len_head_names = len(head_names)
    # len_head_var_names = len(head_var_names)

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
    
    no_date_iden_list = [i for i in range(655+len(head_names)-2,len(head_names)+655)]
    iden_n_d = 0
    # print(no_date_iden_list)
    # print(len(head_names))
    # print(len(padded))
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
    # plt.imshow(img_2)
    # plt.show() 

    ## Need reshape of img_2 to merge img_1 and img_2
    h= img_2.shape[0]
    w= img.shape[1]
    dim = (h, w)
    img_2_resized = cv2.resize(img_2, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

    ## merge img_1 and img_2_resized
    img_3 = np.concatenate([img_1, img_2_resized], axis=0)
    # plt.imshow(img_3)
    # plt.show()

    #################################
    # cv2.resize(img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    head_iden_list = [i for i in range(555+len(head_names),len(padded)+555)]
    iden_i = 0
    # print(head_iden_list)
    # print(len(head_names))
    # print(len(padded))
    for i in range(len(head_names), len(padded)):
        org_img = padded[i]
        h,w = org_img.shape
        dim=(h,w)
        space_img = addSpace(padded[i], head_iden_list[iden_i])
        resize_space_img=cv2.resize(space_img, dim[::-1], fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        padded[i] = resize_space_img
        iden_i+=1

    ##################################

    ## Merge head_var_names
    im_4 = np.concatenate(padded[len(head_names):], axis=0)
    # plt.imshow(im_4)
    # plt.show()

    ## Merge img_3 and img_4: Final imgage
    final_img = np.concatenate([img_3, im_4], axis=0)
    # plt.imshow(final_img)
    # plt.show()

    # return
    return final_img, head_iden_list, no_date_iden_list


