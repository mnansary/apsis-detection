# -*-coding: utf-8 -
'''
    @author: Tahsin Reasat
    Adoptation:MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import pandas as pd
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt
# %matplotlib inline 
import os 
import cv2
import numpy as np
import sys
import numpy
import scipy.ndimage as sni
from pprint import pprint

#--------------------
# Parser class
#--------------------
class GraphemeParser():
    def __init__(self,class_map_csv):
        # gets class map
        self.class_map_csv=class_map_csv
        # initializes components
        self.__getComps()
    
    def __getComps(self):
        '''
            **Private Initialization**

            reads and creates dataframe for roots,consonant_diacritic,vowel_diacritic and graphemes 
            args:
                class_map_csv        : path of classes.csv
            returns:
                tuple(df_root,df_vd,df_cd)
                df_root          :     dataframe for grapheme roots
                df_vd            :     dataframe for vowel_diacritic 
                df_cd            :     dataframe for consonant_diacritic
                
        '''
        # read class map
        df_map=pd.read_csv(self.class_map_csv)
        # get grapheme roots
        df_root = df_map.groupby('component_type').get_group('grapheme_root')
        df_root.index = df_root['label']
        df_root = df_root.drop(columns = ['label','component_type'])
        # get vowel_diacritic
        df_vd = df_map.groupby('component_type').get_group('vowel_diacritic')
        df_vd.index = df_vd['label']
        df_vd = df_vd.drop(columns = ['label','component_type'])
        # get consonant_diacritic
        df_cd = df_map.groupby('component_type').get_group('consonant_diacritic')
        df_cd.index = df_cd['label']
        df_cd = df_cd.drop(columns = ['label','component_type'])
        
        self.vds    =df_vd.component.tolist()
        self.cds    =df_cd.component.tolist()
        self.roots  =df_root.component.tolist()

        

    def word2grapheme(self,word):
        graphemes = []
        grapheme = ''
        i = 0
        while i < len(word):
            grapheme += (word[i])
            # print(word[i], grapheme, graphemes)
            # deciding if the grapheme has ended
            if word[i] in ['\u200d', '্']:
                # these denote the grapheme is contnuing
                pass
            elif word[i] == 'ঁ':  
                # 'ঁ' always stays at the end
                graphemes.append(grapheme)
                grapheme = ''
            elif word[i] in list(self.roots) + ['়']:
                # root is generally followed by the diacritics
                # if there are trailing diacritics, don't end it
                if i + 1 == len(word):
                    graphemes.append(grapheme)
                elif word[i + 1] not in ['্', '\u200d', 'ঁ', '়'] + list(self.vds):
                    # if there are no trailing diacritics end it
                    graphemes.append(grapheme)
                    grapheme = ''

            elif word[i] in self.vds:
                # if the current character is a vowel diacritic
                # end it if there's no trailing 'ঁ' + diacritics
                # Note: vowel diacritics are always placed after consonants
                if i + 1 == len(word):
                    graphemes.append(grapheme)
                elif word[i + 1] not in ['ঁ'] + list(self.vds):
                    graphemes.append(grapheme)
                    grapheme = ''

            i = i + 1
            # Note: df_cd's are constructed by df_root + '্'
            # so, df_cd is not used in the code

        return graphemes

    

def createPrintedLine(iden,
                       comps,
                       font_path,
                       font_size):
    '''
        creates printed word image
        args:
            iden    :       identifier marking value starting
            comps   :       the list of components
            font_path:      the desired font path 
            font_size:      the size of the font
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
    val_offset=iden
    comps=[str(comp) for comp in comps]
    # max dim
    min_offset=100
    max_dim=len(comps)*font_size+min_offset
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    for idx,comp in enumerate(comps):
        if idx < len(comps)-1 and comps[idx+1] in mods:
            comps[idx]+=comps[idx+1]
            comps[idx+1]=None 
            
    comps=[comp for comp in comps if comp is not None]
    word="".join(comps)
    # font path
    font=ImageFont.truetype(font_path, size=font_size)
    # sizes of comps
    (comp_size,_),(_,_) = font.font.getsize(word)  
    # construct labels
    label={}
    imgs=[]
    x=0
    y=0
    comp_str=''
    for comp in comps:
        comp_str+=comp
        # # calculate increment
        # (comp_width,_),(offset,_)=comp_size
        # dx = comp_width+offset 
        # draw
        image = Image.new(mode='L', size=(max_dim,max_dim))
        draw = ImageDraw.Draw(image)
        #draw.text(xy=(x, y), text=comp, fill=iden, font=font)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        
        imgs.append(np.array(image))
        
        # x+=dx
        # label
        label[iden] = comp 
        iden+=1
        
        
    # add images
    img=sum(imgs)
    #img=stripPads(img,0)
    img=img[~np.all(img == 0, axis=1)]
    img=img[:,:comp_size]
    img[img>0]+=val_offset-1
    return img,label,iden


def padImg(line_img,h_max,w_max): ### <<<<<=================
    # shape
    h,w=line_img.shape
    # pad widths
    left_pad_width =(w_max-w)//2
    # print(left_pad_width)
    right_pad_width=w_max-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width))
    right_pad=np.zeros((h,right_pad_width))
    # pad
    line_img =np.concatenate([left_pad,line_img,right_pad],axis=1)
    
    # shape
    h,w=line_img.shape
    # pad heights
    top_pad_height =(h_max-h)//2
    bot_pad_height=h_max-h-top_pad_height
    # pads
    top_pad =np.zeros((top_pad_height,w))
    bot_pad=np.zeros((bot_pad_height,w))
    # pad
    line_img =np.concatenate([top_pad,line_img,bot_pad],axis=0)
    return line_img

def createTable(num_rows,    # indicates product names
                num_columns, # indicates memo headers
                line_width,  
                cell_widths,  
                cell_height):
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



def selectPrintTextRegions(mask, 
                           cell_widths): 
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



def drawsPrintTextOnTable(printTextData,
                          total,
                          class_map_csv_path,
                          font_path,
                          font_size=128,
                          bangla_num=["০","১","২","৩","৪","৫","৬","৭","৮","৯"],
                          iden=3,
                          h_max=160,
                          line_width=2
                          ):

    """
          @function author:
          
          Place Print Bangla text on specific loactions of Binary Image.

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
              
          returns:
              TableImg            =  Binary Image after placing text on desired locations.

    """

    gp=GraphemeParser(class_map_csv_path)           ######## <<<=============== Call GraphemeParser Class

    ## Create Function: for Process the Text data
    def processText(dataText,
                    gp=gp,
                    bangla_num=bangla_num
                    ):
        data=[]
        for p in dataText:
          words=p.split()
          _data=[]
          # add space after a word
          for word in words:
            if any(char in bangla_num for char in word):
              graphemes=[g for g in word]
            else:
              graphemes=gp.word2grapheme(word)
            graphemes.append(" ")
            _data+=graphemes
          data.append(_data)
        return data

    ## Create Function: find images and labels
    def imgsLabels(processTextData,
                   iden=iden,
                   font_path=font_path,
                   font_size=font_size,
                   h_max=h_max
                   ):
        imgs=[]
        labels=[]
        w_max=0

        # find images and labels
        for line in processTextData:                     
          img,label,iden=createPrintedLine(iden,
                                          line,
                                          font_path,             ######## <<<=============== Call Function
                                          font_size)
          h,w=img.shape
          if w>w_max:
            w_max=w

          imgs.append(img)
          labels.append(label)


        w_max+=32
        padded=[]
        for img in imgs:
          img=padImg(img,h_max,w_max) ### <<<<<================= Call Function
          padded.append(img)

        return imgs, labels, padded, w_max

    ### ===== Process printTextData =========== ###
    printTextDataProcess = {}

    ### Process Sl, Brand, Quantity, Rate, Taka Columns
    for key, data_text in printTextData.items():
      data = processText(data_text)                        ######## <<<=============== Call Function
      imgs, labels, padded, w_max = imgsLabels(data)       ######## <<<=============== Call Function
      printTextDataProcess[key] = [imgs, labels, padded, w_max]

    #### =Process Total box  ==== 
    if total:
      data_total=processText(total)                ######## <<<=============== Call Function
      _w_max = list(printTextDataProcess.keys())[-2]
      w_max_total = printTextDataProcess[_w_max][3]
      imgs_total = []
      labels_total = []
      for line in data_total:
         img,label,iden=createPrintedLine(iden,
                                          line,
                                          font_path,             ######## <<<=============== Call Function
                                          font_size)
         imgs_total.append(img)
         labels_total.append(label)         

      padded_total=[]
      for img in imgs_total:
        img=padImg(img,h_max,w_max_total) ### <<<<<================= Function
        padded_total.append(img)
        # plt.imshow(padImg(img,h_max,w_max_total))
        # plt.show()

      printTextDataProcess['total'] = [imgs_total, labels_total, padded_total, w_max_total]

    ## ==== Create Table =========
    _serial_key = list(printTextData.keys())[0]
    num_rows = len(printTextData[_serial_key])+2
    num_columns = len(printTextData)+1 #6/5/7
    cell_height = h_max+line_width
    cell_widths = []
    for key, data_text in printTextData.items():
      w_max = printTextDataProcess[key][3]
      cell_width = w_max + line_width
      cell_widths.append(cell_width)

    ## Create Table
    TableImg=createTable( num_rows,    
                      num_columns, 
                      line_width,  
                      cell_widths,  
                      cell_height)


    ## merge all padded
    padded_all = []
    for key, data_text in printTextDataProcess.items():
      # print(key)
      pad = printTextDataProcess[key][2]
      for k in pad:
        padded_all.append(k)
        # plt.imshow(k)
        # plt.show()

    ## select locations for placing text
    serial_locs,brand_locs,total_loc,other_locs,all_locs,labeled_img=selectPrintTextRegions(TableImg, cell_widths)

    ## Binary Image: TableImg
    TableImg=255-TableImg
    TableImg[TableImg>0]=1

    ## Place text on specific location of TableImg
    for locs, pads in zip(all_locs, padded_all):
      for i,img in zip([locs],[pads]):
        idx = np.where(labeled_img==i)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        TableImg[y_min:y_max,x_min:x_max]=img

        # plt.imshow(TableImg)
        # plt.show() 

    return TableImg, all_locs, labeled_img 


