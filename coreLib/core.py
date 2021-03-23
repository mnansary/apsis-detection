# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
tqdm.pandas()
#--------------------
# config for data gen
#--------------------
class CONFIG:
    # number of lines per image
    MIN_NUM_LINES   =   1
    MAX_NUM_LINES   =   15
    # number of words per line
    MIN_NUM_WORDS   =   1
    MAX_NUM_WORDS   =   5
    # word lenght
    MIN_WORD_LEN    =   1
    MAX_WORD_LEN    =   10
    # word lenght
    MIN_NUM_LEN     =   1
    MAX_NUM_LEN     =   5
    # char height
    SYM_HEIGHT      =   32
    DATA_DIM        =   512
    
    # csv paths for datset core elements
    GRAPHEME_CSV    =   os.path.join(os.getcwd(),"resources","graphemes.csv")
    NUMBER_CSV      =   os.path.join(os.getcwd(),"resources","numbers.csv")   
    # separator paths
    SEPARATORS      =   [_path for _path in glob(os.path.join(os.getcwd(),"resources","separators","*.*"))]
    # margin space
    VERT_MIN_SPACE  =   15
    VERT_MAX_SPACE  =   50
    HORZ_MIN_SPACE  =   15
    HORZ_MAX_SPACE  =   50
    # wrapping
    MAX_WARP_PERC   =   10
# data frames
num_df  =   pd.read_csv(CONFIG.NUMBER_CSV)
char_df =   pd.read_csv(CONFIG.GRAPHEME_CSV)
#--------------------
# helpers
#--------------------
#--------------------------------------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr


#--------------------
# image ops Pure
#--------------------

#--------------------------------------------------------------------------------------------
def getSymImg(img_path,Type=None):
    '''
        cleans and resizes the image after stripping
        args:
            img         :   numpy array grayscale image
        returns:
            resized clean image
    '''
    img=cv2.imread(img_path,0)
    # inversion check for numbers
    if Type=="num":
        basename=os.path.basename(img_path)
        if "e" in basename:
            img=255-img
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # strip
    img=stripPads(arr=img,val=255)
    # resize to char dim
    h,w=img.shape 
    img_width= int(CONFIG.SYM_HEIGHT* w/h) 
    img=cv2.resize(img,(img_width,CONFIG.SYM_HEIGHT),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # invert
    img=255-img
    return img

#--------------------
# image ops Synthetic
#--------------------
def createNumberImage(num_len,raw_nums_path,iden_val):
    '''
        creates a number image based on given lenght
        args:
            num_len         : length of the number
            raw_nums_path   : directory that contains the raw number images
            iden_val        : the indentifier-value to be assigned to the image
            
    '''
    data=[]
    # number 
    num_data= num_df.sample(n = num_len)
    # create images
    for idx,row in num_data.iterrows():
        _file           =   row["filename"]
        label           =   row["digit"]
        img_path        =   os.path.join(raw_nums_path,_file)
        img             =   getSymImg(img_path,Type="num")
        img[img>0]      =   iden_val
        data.append([img,str(label),iden_val])
        iden_val+=1
    return data ,iden_val

def createDateImage(year_len,raw_nums_path,iden_val):
    '''
        creates the date image
        args:
            year_len        :   length of the year digit
            raw_nums_path   : directory that contains the raw number images
            iden_val        : the indentifier-value to be assigned to the image
            

    '''
    
    day_data,iden_val        = createNumberImage(2,raw_nums_path,iden_val)
    img                      = getSymImg(random.choice(CONFIG.SEPARATORS))
    img[img>0]               = iden_val
    separator_data_1         = [[img,"/",iden_val]]
    iden_val+=1

    month_data,iden_val      = createNumberImage(2,raw_nums_path,iden_val)
    img                      = getSymImg(random.choice(CONFIG.SEPARATORS))
    img[img>0]               = iden_val
    separator_data_2         = [[img,"/",iden_val]]
    iden_val+=1
    
    year_data,iden_val       = createNumberImage(year_len,raw_nums_path,iden_val)
    return  day_data+separator_data_1+month_data+separator_data_2+year_data,iden_val    

def createWordImage(word_len,raw_path,iden_val):
    '''
        creates a word image based on given lenght
        args:
            word_len   : length of the word
            raw_path   : directory that contains the raw grapheme images
            iden_val   : the indentifier-value to be assigned to the image
            
    '''
    data=[]
    # number 
    char_data= char_df.sample(n = word_len)
    # create images
    for idx,row in char_data.iterrows():
        _file           =   row["image_id"]
        label           =   row["grapheme"]
        img_path        =   os.path.join(raw_path,f"{_file}.png")
        img             =   getSymImg(img_path)
        img[img>0]      =   iden_val
        data.append([img,label,iden_val])
        iden_val+=1
    return data ,iden_val


#--------------------
# dataset ops 
#--------------------
def padLineLeftRight(max_line_width,line_img):
    '''
        pads an image left and right
        args:
            max_line_width  : width of the max line length
            line_img        : image to pad
    '''
    # shape
    h,w=line_img.shape
    # pad widths
    left_pad_width =random.randint(0,(max_line_width-w))
    right_pad_width=max_line_width-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width),dtype=np.int64)
    right_pad=np.zeros((h,right_pad_width),dtype=np.int64)
    # pad
    line_img =np.concatenate([left_pad,line_img,right_pad],axis=1)
    return line_img

def createLabeledImage(raw_path,
                       raw_nums_path,
                       nimg):
    '''
        takes the config defined as the base parameter 
        args:
            raw_path        : directory that contains the raw grapheme images
            raw_nums_path   : directory that contains the raw number images
            nimg            : the identifier of an image
    '''
    
    iden_val=1
    
    page_anon=[]
    page_parts=[]
    # select number of lines in an image
    num_lines=random.randint(CONFIG.MIN_NUM_LINES,
                             CONFIG.MAX_NUM_LINES)
    
    for nl in range(num_lines):
        line_parts=[]
        # select number of words
        num_words=random.randint(CONFIG.MIN_NUM_WORDS,
                                 CONFIG.MAX_NUM_WORDS)
        _types=random.choices(population=["number", "date", "word"],
                              weights=[0.2, 0.1, 0.7],
                              k=num_words)
        for tidx,_type in enumerate(_types):
            if _type=="word":
                word_len=random.randint(CONFIG.MIN_WORD_LEN,
                                        CONFIG.MAX_WORD_LEN)
                _data,iden_val=createWordImage(word_len,raw_path,iden_val)
            elif _type=="number":
                number_len=random.randint(CONFIG.MIN_NUM_LEN,
                                          CONFIG.MAX_NUM_LEN)
                _data,iden_val=createNumberImage(number_len,raw_nums_path,iden_val)
            elif _type=="date":
                year_len=random.choice([2,4])
                _data,iden_val=createDateImage(year_len,raw_nums_path,iden_val)
            
            # create the part: word number etc image
            part_imgs=[]
            for part in _data:
                _img   = part[0]
                _label = part[1]
                _iden  = part[2]
                _line  = nl
                _img_id= nimg
                page_anon.append({"ImageId":_img_id,
                                  "LineNumber":_line,
                                  "IdenValue":_iden,
                                  "Label":_label})
                part_imgs.append(_img)
            
            # with out the last part all parts are paded
            if tidx <len(_types)-1:
                # add pad 
                part_pad=np.ones((CONFIG.SYM_HEIGHT,
                                  random.randint(CONFIG.HORZ_MIN_SPACE,
                                                 CONFIG.HORZ_MAX_SPACE)),dtype=np.int64)*iden_val
                page_anon.append({"ImageId":_img_id,
                                  "LineNumber":_line,
                                  "IdenValue":iden_val,
                                  "Label":" "})
                # increase iden
                iden_val+=1
                # complete part img    
                part_imgs.append(part_pad)
            # the part image
            part_img=np.concatenate(part_imgs,axis=1)
            # append
            line_parts.append(part_img)
        # create the line image
        line_img=np.concatenate(line_parts,axis=1)
        # the page lines
        page_parts.append(line_img)
    
    
    # find max line width
    max_line_width=0
    for line in page_parts:
        _,w=line.shape
        if w>=max_line_width:
            max_line_width=w
            
    # pad each line to max_width
    paded_parts=[]
    for lidx,line_img in enumerate(page_parts):
        line_img=padLineLeftRight(max_line_width,line_img)
        # top pad for first one
        if lidx==0:
            pad_height_top=random.randint(CONFIG.VERT_MIN_SPACE,
                                          CONFIG.VERT_MAX_SPACE*2)
            pad_top=np.zeros((pad_height_top,max_line_width))
            line_img=np.concatenate([pad_top,line_img],axis=0)
        # pad lines 
        pad_height=random.randint(CONFIG.VERT_MIN_SPACE,
                                  CONFIG.VERT_MAX_SPACE)
        pad     =np.zeros((pad_height,max_line_width))
        line_img=np.concatenate([line_img,pad],axis=0)
        paded_parts.append(line_img)
    # page img
    page_img=np.concatenate(paded_parts,axis=0)
    # page anon
    page_anon=pd.DataFrame(page_anon)
    # add coords
    COORDS=[]
    for i in page_anon.IdenValue.tolist():
        try:
            idx = np.where(page_img==i)
            y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            x1,x2,x3,x4=x_min,x_max,x_max,x_min
            y1,y2,y3,y4=y_min,y_min,y_max,y_max
            coords=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype="float32")
            COORDS.append(coords)
        except Exception as e:
            print(i,e)
    page_anon["Coords"]=COORDS
    # shape image
    
    df_space=page_anon.loc[page_anon.Label==' ']
    space_vals=df_space.IdenValue.tolist()
    for sv in space_vals:
        page_img[page_img==sv]=0
    page_img[page_img>0]=255
    page_img=page_img.astype("uint8")
    page_img=255-page_img
    # transformation
    # gets height and width
    height,width=page_img.shape
    # location of source
    src    = np.float32([[0,0], 
                         [width-1,0], 
                         [width-1,height-1], 
                         [0,height-1]])
    # construct destination
    left_warp =int((CONFIG.DATA_DIM*random.randint(0,CONFIG.MAX_WARP_PERC) )/100)
    right_warp=CONFIG.DATA_DIM-int((CONFIG.DATA_DIM*random.randint(0,CONFIG.MAX_WARP_PERC) )/100)
    dst    = np.float32([[left_warp,0], 
                         [right_warp,0], 
                         [CONFIG.DATA_DIM-1,CONFIG.DATA_DIM-1], 
                         [0,CONFIG.DATA_DIM-1]])
    M   = cv2.getPerspectiveTransform(src, dst)
    page_img = cv2.warpPerspective(page_img, M, (CONFIG.DATA_DIM,
                                          CONFIG.DATA_DIM))
    
    # co-ords
    TRANSFORMED=[]
    for coords in page_anon.Coords.tolist():
        TRANSFORMED.append(cv2.perspectiveTransform(src=coords[np.newaxis], m=M)[0])
    # df
    page_anon["data"]=TRANSFORMED
    return page_img,page_anon         