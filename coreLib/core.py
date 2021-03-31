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
import math

from shapely import geometry
from scipy import spatial
from tqdm import tqdm
from glob import glob
tqdm.pandas()
#--------------------
# config for data gen
#--------------------
class CONFIG:
    # number of lines per image
    MIN_NUM_LINES   =   1
    MAX_NUM_LINES   =   7
    # number of words per line
    MIN_NUM_WORDS   =   1
    MAX_NUM_WORDS   =   5
    # word lenght
    MIN_WORD_LEN    =   1
    MAX_WORD_LEN    =   8
    # num lenght
    MIN_NUM_LEN     =   1
    MAX_NUM_LEN     =   4
    # char height
    SYM_HEIGHTS     =   [32+i for i in range(0,64,8)]
    SYM_HEIGHT      =   32
    DATA_DIM        =   512
    
    # csv paths for datset core elements
    GRAPHEME_CSV    =   os.path.join(os.getcwd(),"resources","graphemes.csv")
    NUMBER_CSV      =   os.path.join(os.getcwd(),"resources","numbers.csv")   
    # separator paths
    SEPARATORS      =   [_path for _path in glob(os.path.join(os.getcwd(),"resources","separators","*.*"))]
    # margin space
    VERT_MIN_SPACE  =   30
    VERT_MAX_SPACE  =   100
    HORZ_MIN_SPACE  =   60
    HORZ_MAX_SPACE  =   100
    # wrapping
    MAX_WARP_PERC   =   20
    # rotation 
    MAX_ROTATION    =   40
# data frames
num_df  =   pd.read_csv(CONFIG.NUMBER_CSV)
char_df =   pd.read_csv(CONFIG.GRAPHEME_CSV)
#-------------------
# global ops
#-------------------
def get_gaussian_heatmap(size=512, distanceRatio=1.5):
    '''
        creates a gaussian heatmap
        This is a fixed operation to create heatmaps
    '''
    # distrivute values
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    # create a value mesh grid
    x, y = np.meshgrid(v, v)
    # spreading heatmap
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')
# fixed heatmap
heatmap_text=get_gaussian_heatmap(size=CONFIG.DATA_DIM,distanceRatio=3.5)
heatmap_link=get_gaussian_heatmap(size=CONFIG.DATA_DIM,distanceRatio=1.5)

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
        
        # select a random height
        CONFIG.SYM_HEIGHT=random.choice(CONFIG.SYM_HEIGHTS)
                
        for tidx,_type in enumerate(_types):
            if _type=="word":
                # select a random length
                word_len=random.randint(CONFIG.MIN_WORD_LEN,
                                        CONFIG.MAX_WORD_LEN)
                # create data                        
                _data,iden_val=createWordImage(word_len,raw_path,iden_val)
            elif _type=="number":
                # select a random length
                number_len=random.randint(CONFIG.MIN_NUM_LEN,
                                          CONFIG.MAX_NUM_LEN)
                # create data
                _data,iden_val=createNumberImage(number_len,raw_nums_path,iden_val)
            elif _type=="date":
                # select a random length
                year_len=random.choice([2,4])
                # create data
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
            
            # all parts are paded
            if tidx <len(_types):
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
    h_pad,_=page_img.shape
    _pad=np.zeros((h_pad,random.randint(CONFIG.VERT_MIN_SPACE,CONFIG.VERT_MAX_SPACE*2)),dtype=np.int64)
    page_img=np.concatenate([_pad,page_img,_pad],axis=1)
    
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


    # scale transformation
    (nH, nW) = page_img.shape
    # rescale
    page_img=cv2.resize(page_img,(CONFIG.DATA_DIM,CONFIG.DATA_DIM))

    rx=CONFIG.DATA_DIM/nW
    ry=CONFIG.DATA_DIM/nH
    # change coords
    TRANSFORMED=[]
    for coords in page_anon.Coords:
        coords[:,0]=coords[:,0]*rx
        coords[:,1]=coords[:,1]*ry
        TRANSFORMED.append(coords)
    # df
    page_anon["Coords"]=TRANSFORMED
    # sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    page_img = cv2.filter2D(page_img, -1, kernel)
    
    return page_img,page_anon


#--------------------
# post processing ops 
#--------------------

#--------------------------------------------------------------------------------------------


def warpLabeledData(page_img,page_anon):
    '''
        creates a warped Image and corresponding annotation
        args:
            page_img    :   labeled image 
            page_anon   :   annotation
    '''  
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
    page_anon["Coords"]=TRANSFORMED
    return page_img,page_anon         

def rotateLabeledData(page_img,page_anon):
    '''
        creates a warped Image and corresponding annotation
        args:
            page_img    :   labeled image 
            page_anon   :   annotation
    '''  
    # get shape
    (h, w) = page_img.shape
    # get center
    center = (w / 2, h / 2)
    # angle to rotate
    angle  = random.randint(-CONFIG.MAX_ROTATION,CONFIG.MAX_ROTATION)
    # get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosM = np.abs(M[0][0])
    sinM = np.abs(M[0][1])
    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    nH = int((h * sinM) +(w * cosM))
    nW = int((h * cosM) +(w * sinM))
 
    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    M[0][2] += (nW/2) - center[0]
    M[1][2] += (nH/2) - center[1]
    # rotate
    page_img = cv2.warpAffine(page_img, M, (nW,nH))
    
    
    
    # co-ords
    TRANSFORMED=[]
    for coords in page_anon.Coords.tolist():
        new_coords=[]
        coord_mat=np.concatenate([coords,np.ones((4,1))],axis=1)
        for c in coord_mat:
            new_coords.append(np.dot(M,c))
        TRANSFORMED.append(np.array(new_coords))
        
    # df
    page_anon["Coords"]=TRANSFORMED
    
    
    # scale transformation
    (nH, nW) = page_img.shape

    # rescale
    page_img=cv2.resize(page_img,(CONFIG.DATA_DIM,CONFIG.DATA_DIM))

    rx=CONFIG.DATA_DIM/nW
    ry=CONFIG.DATA_DIM/nH
    # change coords
    TRANSFORMED=[]
    for coords in page_anon.Coords:
        coords[:,0]=coords[:,0]*rx
        coords[:,1]=coords[:,1]*ry
        TRANSFORMED.append(coords)
    # df
    page_anon["Coords"]=TRANSFORMED
    
    
    return page_img,page_anon



def blurrImg(img):
    '''
        blurs an image
    '''
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)
def addNoise(img):
    '''
        adds gaussian noise to image
    '''
    row, col, _ = img.shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    # create noisy image
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    img = img*0.75+gaussian*0.25
    img=img.astype("uint8")
    return img

#--------------------
# image ops main
#--------------------
#--------------------------------------------------------------------------------------------

def createSingleImage(raw_path,
                       raw_nums_path,
                       nimg):
    '''
        takes the config defined as the base parameter 
        args:
            raw_path        : directory that contains the raw grapheme images
            raw_nums_path   : directory that contains the raw number images
            nimg            : the identifier of an image
    '''
    # create base img
    img,df=createLabeledImage(raw_path,raw_nums_path,nimg)
    # post processing ops
    if random.choices(population=[0,1],weights=[0.3, 0.7],k=1)[0]==1:
        img,df=warpLabeledData(img,df)
    if random.choices(population=[0,1],weights=[0.3, 0.7],k=1)[0]==1:
        img,df=rotateLabeledData(img,df)
    # three channel conversion    
    img=np.expand_dims(img,axis=-1)
    img=np.concatenate([img,img,img],axis=-1)
    # gen ops
    if random.choices(population=[0,1],weights=[0.5, 0.5],k=1)[0]==1:
        img=addNoise(img)
    if random.choices(population=[0,1],weights=[0.5, 0.5],k=1)[0]==1:
        img=blurrImg(img)
    return img,df
#--------------------
# dataset OPS
#--------------------
#--------------------------------------------------------------------------------------------
#--------------------
# dataset helpers
#--------------------

def constructLine(df):
    '''
        convert the dataframe into lines
        args:
            df  :   page_annotation dataframes
    '''
    eng_nums=['0','1','2','3','4','5','6','7','8','9']
    ban_nums=['০','১','২','৩','৪','৫','৬','৭','৮','৯']
    lines=[]
    for lineNum in df.LineNumber.tolist():
        linedf=df.loc[df.LineNumber==lineNum]
        line=[]
        for label,coords in zip(df.Label.tolist(),df.Coords.tolist()):
            if label in eng_nums:
                idx=eng_nums.index(label)
                label=ban_nums[idx]
            line.append((coords,label))
        lines.append(line)
    return lines

def get_rotated_box(points):
    """
        Obtain the parameters of a rotated box.
        Returns:
            The vertices of the rotated box in top-left,
            top-right, bottom-right, bottom-left order along
            with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation




#--------------------
# dataset maps
#--------------------
def compute_maps(lines):
    '''
        creates a textmap and a link map according to the paper
        args:
            heatmap         :   base gaussian heatmap
            image_height    :   height of the image
            image_width     :   width of the image
            lines           :   tuples of (coords,labels) as line list    
    '''
    # image size
    image_height    = CONFIG.DATA_DIM
    image_width     = CONFIG.DATA_DIM  
    
    textmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')
    linkmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')
    # use text heatmap as base
    heatmap         = heatmap_text    
    # source bbobx of heatmap
    src = np.array([[0, 0], 
                    [heatmap.shape[1], 0], 
                    [heatmap.shape[1], heatmap.shape[0]],
                    [0, heatmap.shape[0]]]).astype('float32')

    for line in lines:
        # this is fixed for now 
        orientation='horizontal'
        previous_link_points = None
        
        for [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], c in line:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(lambda v: max(v, 0),[x1, y1, x2, y2, x3, y3, x4, y4])
            # space indicates the end of a line/word
            if c == ' ':
                previous_link_points = None
                continue
            yc = (y4 + y1 + y3 + y2) / 4
            xc = (x1 + x2 + x3 + x4) / 4
            if orientation == 'horizontal':
                current_link_points = np.array([[(xc + (x1 + x2) / 2) / 2, (yc + (y1 + y2) / 2) / 2], [(xc + (x3 + x4) / 2) / 2, (yc + (y3 + y4) / 2) / 2]]) / 2
            else:
                current_link_points = np.array([[(xc + (x1 + x4) / 2) / 2, (yc + (y1 + y4) / 2) / 2], [(xc + (x2 + x3) / 2) / 2, (yc + (y2 + y3) / 2) / 2]]) / 2
            character_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype('float32') / 2
            # check for linking
            if previous_link_points is not None:
                if orientation == 'horizontal':
                    link_points = np.array([previous_link_points[0], current_link_points[0], current_link_points[1],previous_link_points[1]])
                else:
                    link_points = np.array([previous_link_points[0], previous_link_points[1], current_link_points[1],current_link_points[0]])
                # transforms the bbox and creates linkmap
                ML = cv2.getPerspectiveTransform(src=src,dst=link_points.astype('float32'),)
                linkmap += cv2.warpPerspective(heatmap_link,ML,dsize=(linkmap.shape[1],linkmap.shape[0])).astype('float32')
            # transforms the bbox and creates the heatmap
            MA = cv2.getPerspectiveTransform(src=src,dst=character_points)
            textmap += cv2.warpPerspective(heatmap_text, MA, dsize=(textmap.shape[1],textmap.shape[0])).astype('float32')
            # new linking points
            previous_link_points = current_link_points
            
    return textmap.clip(0,255),linkmap.clip(0,255)


#--------------------
# dataset OPS wrapper
#--------------------
#--------------------------------------------------------------------------------------------
def create_single_data(raw_path,
                       raw_nums_path,
                       nimg):
    '''
    takes the config defined as the base parameter 
        args:
            raw_path        : directory that contains the raw grapheme images
            raw_nums_path   : directory that contains the raw number images
            nimg            : the identifier of an image
    '''
    # get image and data
    img,df=createSingleImage(raw_path,raw_nums_path,nimg)
    # get lines
    lines=constructLine(df)
    # get textmap and linkmap
    textmap,linkmap=compute_maps(lines)
    return img,df,textmap,linkmap