# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np 
import cv2 
import random
from glob import glob
import os
import scipy.ndimage as sni
from .word import createPrintedLine, processLine,imgsLabels, create_word
from .config import config
from .utils import padImg,rotate_image

#----------------------------
# table function
#----------------------------
def createTable(num_rows, num_columns, line_width, cell_widths, cell_height):

    '''
        @function author:
        
        creates a binary image where 0 values indicate table lines and 255 indicates background
        args:
            num_rows    =   the desired number of rows in the table             : type <int>
                            * this indicates the number of products in the memo
                            * the last row may only have two columns:
                                * TOTAL: amount in numbers
                                
                            
            num_columns =   the desired number of columns in the table          : type <int>
                            * can be only 5 or 6 
                            * these are the memo headers like:
                                * serial
                                * brand/product
                                * amount 
                                * rate
                                * total
                                * description
                            
            line_width  =   the width of the line of the table                  : type <int>
            
            cell_widths =   the width of a single cell                          : type <LIST>                   
                            * for each column a width must be provided 
            
            cell_height =   the height of a single cell                         : type <int>
            
        returns:
            table_mask  =   the binary mask
            (MUST BE IN BINARY ,SINCE WE WILL BE PLACING MARKED TEXTS WITHIN THE CELL)
        
        algorithm (simplified):
            * for number of rows+1 , 
                draw lines (having the line width) 
                which are [cell_height] apart from one another
            * for number of columns+1 ,
                draw lines (having the line width)
                which are [cell_width] apart from one another
        tools:
            * cv2.line
            * rgb mode execution example: 
            https://study.marearts.com/2018/11/python-opencv-draw-grid-example-source.html
            
    '''
    # error check for types and values
    assert type(num_rows)==int and num_rows>0, "Wrong value for:num_rows"
    assert type(num_columns)==int and num_columns>0, "Wrong value for:num_columns"
    assert type(line_width)==int and line_width>0,"Wrong value for:line_width"
    assert type(cell_height)==int and cell_height>0,"Wrong value for:cell_height"
    assert type(cell_widths)==list and len(cell_widths)==num_columns-1, "For each column a width must be provided"
    for cell_width in cell_widths:
        assert type(cell_width)==int and cell_width>0,"Wrong value for:cell_widths in list"
    # create the mask
    ## calculate the height of the mask
    mask_height =   num_rows*(line_width+cell_height)+line_width
    ## calculate the width of the mask
    mask_width  =   sum([line_width+cell_width for cell_width in cell_widths])+line_width
    ## table mask
    table_mask  =   np.zeros((mask_height,mask_width))
    
    # calculate x,y for rows and columns
    #...........................complete the rest of it............................
    ## check how to draw cv.line with grayscale values 
    ## keep in mind that the (x1,y1) and (x2,y2) must account for line width 
    ## comment each major step for every sub-step

    ## white background
    table_mask = np.full(table_mask.shape, 255, dtype=np.uint8)  # white: 255, black: 0

    ## calculate x column lines
    x = [0]
    a = 0
    for cell_width in cell_widths:
        a += cell_width+line_width
        x.append(a)

    ## calculate y for row lines
    y = [0]
    b = 0
    for i in range(num_rows):
        b+= cell_height+line_width
        y.append(b)


    ## calculate box for vertical lines
    v_xy = []
    for i in range(num_columns):
        if i==0 or i == num_columns-3 or i == num_columns-2 or i == num_columns-1:
            v_xy.append( [int(x[i]), 0, int(x[i]), mask_height-int(cell_height+line_width)] )
        else:
            v_xy.append( [int(x[i]), 0, int(x[i]), mask_height-int(cell_height+line_width)*2 ] )

    ## calculate box for horizontal lines
    h_xy = []
    for i in range(num_rows):
        h_xy.append( [0, int(y[i]), mask_width, int(y[i])] )

    ## draw box on the images
    for val in v_xy:
        [x1, y1, x2, y2] = val 
        cv2.line(table_mask, (x1,y1), (x2, y2), 0, line_width )
    for val in h_xy:
        [x1_, y1_, x2_, y2_] = val 
        cv2.line(table_mask, (x1_,y1_), (x2_, y2_), 0,  line_width )

    return table_mask


def selectPrintTextRegions(mask, cell_widths): 
    '''
          @function author:
          
          select the regions for placing text on mask images.

          args:
              mask        =   table image which contains raws and columns  
              
              cell_widths =   the width of a single cell                          : type <LIST>                   
                              * for each column a width must be provided 
              
          returns:
              slt_serial  =    select label regions of SL raws : type <LIST>
              slt_brand   =    select label regions of Brand raws : type <LIST>
              slt_total   =    select label regions of Total : type <int>  
              slt_others  =    select label regions of other columns name i.e., Quality, Rate, Taka : type <LIST>
              final_list  =    merge of all lists (slt_serial + slt_brand + [slt_total] + slt_others) : type <LIST>
              label_img   =    an integer 'ndarray' where each unique feature in input has a unique label in the returned array.
          
          tools:
              * cv2
              * scipy.ndimage.measurements.label
              https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.measurements.label.html
              
      '''

    '''
        label_img  =  An integer 'ndarray' where each unique feature in input has a unique label in the returned array.
        num_areas  =  'int', then it will be updated with values in labeled_array and only num_features will be returned by this function.
    '''
    label_img,num_areas = sni.measurements.label(mask)
    # unique value (array) of label_img
    lst = np.unique(label_img)[1:num_areas]
    # length of 'cell_widths' param
    len_cell_widths=len(cell_widths)
    
    # store unique value for Sl columns
    slt_serial = []
    for val in lst:
      if val%len_cell_widths == 1:
        slt_serial.append(val)
    slt_serial = slt_serial[:-1]

    # store unique value for Brand columns
    slt_brand = []
    for val in lst:
      if val%len_cell_widths == 2:
        slt_brand.append(val)
    slt_brand = slt_brand[:-1]

    # store unique value Total box
    slt_total = lst[len(lst)-2]

    # store unique value for other columns
    slt_others = list(range(3,len_cell_widths+1))

    # merge of all lists
    final_list = slt_serial + slt_brand + slt_others + [slt_total]

    return slt_serial, slt_brand, slt_total, slt_others, final_list,label_img


def drawsPrintTextOnTable(ds, printTextData, total):

    """
          @function author:
          
          Place Print Bangla text on specific loactions of Binary Image.

          args:
              ds              =    dataset object that holds all the paths and resources

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

              
              
          returns:
              TableImg            =  Binary Image after placing text on desired locations.
              all_locs            =  Location of printed text on image 
              labeled_img         =  an integer 'ndarray' where each unique feature in input has a unique label in the returned array.
          

    """

    ## choose fond
    fonts=[_font for _font in  glob(os.path.join( ds.bangla.fonts,"*.ttf")) if "ANSI" not in _font]
    font_path=random.choice(fonts)
    iden = 3

    ### ===== Process printTextData =========== ###
    printTextDataProcess = {}

    ### Process Sl, Brand, Quantity, Rate, Taka Columns
    for key, data_text in printTextData.items():
        data = [processLine(line) for line in data_text] #processText(data_text)                        ######## <<<=============== Call Function
        imgs, labels, padded, w_max = imgsLabels(ds, data)       ######## <<<=============== Call Function
        printTextDataProcess[key] = [imgs, labels, padded, w_max]

    #### =Process Total box  ==== 
    if total:
        data_total=[processLine(line) for line in total] # processText(total)                ######## <<<=============== Call Function
        _w_max = list(printTextDataProcess.keys())[-2]
        w_max_total = printTextDataProcess[_w_max][3]
        imgs_total = []
        labels_total = []
        for line in data_total:
            img,label,iden=createPrintedLine(iden,line,font_path,config.headline3_font_size)
            imgs_total.append(img)
            labels_total.append(label)         

        padded_total=[]
        for img in imgs_total:
            img=padImg(img,config.height_box,w_max_total) ### <<<<<================= Function
            padded_total.append(img)

        printTextDataProcess['total'] = [imgs_total, labels_total, padded_total, w_max_total]

    ## ==== Create Table =========
    _serial_key = list(printTextData.keys())[0]
    num_rows = len(printTextData[_serial_key])+2
    num_columns = len(printTextData)+1 #6/5/7
    cell_height = config.height_box+config.line_width
    cell_widths = []
    for key, data_text in printTextData.items():
        w_max = printTextDataProcess[key][3]
        cell_width = w_max + config.line_width
        cell_widths.append(cell_width)

    ## Create Table
    TableImg=createTable( num_rows,num_columns, config.line_width, cell_widths,  cell_height)

    ## merge all padded
    padded_all = []
    for key, data_text in printTextDataProcess.items():
        # print(key)
        pad = printTextDataProcess[key][2]
        for k in pad:
            padded_all.append(k)

    ## select locations for placing text
    _,_,_,_,all_locs,labeled_img=selectPrintTextRegions(TableImg, cell_widths)

    ## Binary Image: TableImg
    TableImg=255-TableImg
    TableImg[TableImg>0]=1

    ## Place text on specific location of TableImg
    for locs, pads in zip(all_locs, padded_all):
      for i,img in zip([locs],[pads]):
        idx = np.where(labeled_img==i)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        TableImg[y_min:y_max,x_min:x_max]=img

    return TableImg, all_locs, labeled_img 


def placeHandwrittenTextOnTableImg(ds, printTextData, total):
    
    """
          @function author:
          
          Place Print Bangla text and handwritten text on specific loactions of Binary Image.

          args:
              ds                =    DataSet for Symbol (numbers hand written, bangla hand written)

              printTextData     =    format of example dictionary: dict_ = {'serial': ["সেরিঃ",  
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
              
          returns:
            Table_Image_with_Text =  Binary Image after placing pritten text and hand written on desired locations.

    """
    
    
    ## call func "drawsPrintTextOnTable()" for creating table image with text 
    Table_Image_with_Text, all_locs, labeled_img = drawsPrintTextOnTable(ds, printTextData, total)
    
    ## find locations where you want to place handwritten text
    others_locs = [i for i in range(1, all_locs[-1]+2) if i not in all_locs if i != all_locs[len(all_locs)-1]-1]
    
    ## number of columns
    num_cols = len(printTextData)
    
    ## select some locations where you want to place handwritten text
    List = [i for i in range(1,7)]
    N = random.choice(List) # single item from the List
    Updated_others_locs = random.sample(others_locs, N)
    
    ## Check whether you want to place rotate text or straight text
    rotated_check=[bool(i) for i in np.array(np.random.randint(0,2,len(Updated_others_locs)))] # For TRUE
    rot=0
    imgs = []
    for i in range(2, len(Updated_others_locs)+2):
        img,_,_=create_word(i, "bangla", "handwritten", "number", ds=ds, use_dict=True)

        if rotated_check[rot] == True:
            List_degree = [i for i in range(10,45+5,5)]
            N_deg = random.choice(List_degree) # single item from the List
            rotated_img,M=rotate_image(img,N_deg)
            imgs.append(rotated_img)

        else:
            imgs.append(img)

        rot += 1
        
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
    
