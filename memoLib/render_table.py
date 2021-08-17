# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary, MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
from .render import *

#----------------------------
# render capacity: table 
#----------------------------
def renderMemoTable(ds,language):
    """
        @function author:        
        Create image of table part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
               
    """
    #--------------------------------------------
    # resources
    #--------------------------------------------
    
    if language=="bangla":
        graphemes =ds.bangla_graphemes
        numbers   =ds.bangla.number_values
        font_paths=[font_path for font_path in glob(os.path.join(ds.bangla.fonts,"*.*")) if "ANSI" not in font_path and "Lohit" not in font_path]
        g_df     =ds.bangla.graphemes.df 
        n_df     =ds.bangla.numbers.df 

    else:
        graphemes =  list(string.ascii_lowercase)
        numbers   =  [str(i) for i in range(10)]
        font_paths=[font_path for font_path in glob(os.path.join(ds.english.fonts,"*.*"))]
        g_df     =ds.english.graphemes.df 
        n_df     =ds.english.numbers.df 

    sdf         =  ds.common.symbols.df
    nsdf        =  pd.concat([n_df,sdf],ignore_index=True)
    gsdf        =  pd.concat([g_df,sdf],ignore_index=True)
    adf         =  pd.concat([n_df,g_df,sdf],ignore_index=True)
    #--------------------------------------------
    # product
    #--------------------------------------------
    table=Table()
    place=Placement()
    maps=renderFontMaps(table,random.choice(font_paths))
    table=rand_products(graphemes,numbers,table)
    #--------------------------------------------
    # fill-up products
    #--------------------------------------------
    ## image
    h_max=0
    w_max=0
    prod_images=[]
    prod_cmaps=[]
    prod_wmaps=[]
    ## create line sections
    for line_data in table.products:
        assert len(line_data)==1
        data=line_data[0]
        img,cmap,wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        prod_images.append(img)
        prod_cmaps.append(cmap)
        prod_wmaps.append(wmap)
        
        
    
    prod_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_images]
    prod_images=[padAllAround(line_img,table.pad_dim) for line_img in prod_images]
    
    prod_cmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_cmaps]
    prod_cmaps=[padAllAround(line_img,table.pad_dim) for line_img in prod_cmaps]
    
    prod_wmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in prod_wmaps]
    prod_wmaps=[padAllAround(line_img,table.pad_dim) for line_img in prod_wmaps]
    
    
    #--------------------------------------------
    # fill headers
    #--------------------------------------------
    header_images=[]
    header_wmaps=[]
    header_cmaps=[]
    # fixed for all of them now
    font_size   =   data["font_size"]
    cell_height =   prod_images[0].shape[0]
    w_prod      =   prod_images[0].shape[1]
    ##serial
    if language=="bangla":
        word=random.choice(table.serial["bn"])
    else:
        word=random.choice(table.serial["en"])
    img,cmap,wmap=createPrintedLine(word,font=maps[str(font_size)])
    header_images.append(padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim))
    header_cmaps.append(padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim))
    header_wmaps.append(padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim))
    
    ##column headers
    for i in range(random.randint(table.num_extCOL_min,table.num_extCOL_max)):
        num_words=random.choice([1,2])
        _imgs=[]
        _cmaps=[]
        _wmaps=[]
        _hmax=0
        _wmax=0
        for _ in range(num_words):
            word=rand_word(graphemes,None,table.word_len_max,table.word_len_min)
            img,cmap,wmap=createPrintedLine(word,font=maps[str(font_size)])
            _imgs.append(img)
            _cmaps.append(cmap)
            _wmaps.append(wmap)
            _h,_w=img.shape
            if _h>_hmax:_hmax=_h
            if _w:_wmax:_wmax=_w
        
        
        _imgs=[padToFixedHeightWidth(_img,_hmax,_wmax) for _img in _imgs]
        _cmaps=[padToFixedHeightWidth(_img,_hmax,_wmax) for _img in _cmaps]
        _wmaps=[padToFixedHeightWidth(_img,_hmax,_wmax) for _img in _wmaps]
        if len(_imgs)>0:
            img=np.concatenate(_imgs,axis=0)
            cmap=np.concatenate(_cmaps,axis=0)
            wmap=np.concatenate(_wmaps,axis=0)
        else:
            img=_imgs[0]
            cmap=_cmaps[0]
            wmap=_wmaps[0]
        
        _h,_w=img.shape
        if _h>cell_height:
            cell_height=_h

        if i==0:
            # prod column
            img=padToFixedHeightWidth(img,cell_height,w_prod)
            cmap=padToFixedHeightWidth(cmap,cell_height,w_prod)
            wmap=padToFixedHeightWidth(wmap,cell_height,w_prod)
        else:
            img=padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim)
            cmap=padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim)
            wmap=padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim)
        
        header_images.append(img)
        header_cmaps.append(cmap)
        header_wmaps.append(wmap)
        
        

    # fill total
    word=rand_word(graphemes,None,table.word_len_max,table.word_len_min)
    img,cmap,wmap=createPrintedLine(word[:-1],font=maps[str(font_size)])
    
    total_img=padToFixedHeightWidth(img,cell_height,img.shape[1]+2*table.pad_dim)
    total_cmap=padToFixedHeightWidth(cmap,cell_height,img.shape[1]+2*table.pad_dim)
    total_wmap=padToFixedHeightWidth(wmap,cell_height,img.shape[1]+2*table.pad_dim)
    
    # fill serial
    serial_images=[]
    serial_cmaps=[]
    serial_wmaps=[]
    
    serial_width=header_images[0].shape[1]
    for i in range(len(prod_images)):
        sel_val=str(i+1)
        word="".join([v for v in sel_val])
        if language=="bangla":
            word="".join([numbers[int(v)] for v in word])
        
        img,cmap,wmap=createPrintedLine(word,font=maps[str(font_size)])
        serial_images.append(padToFixedHeightWidth(img,cell_height,serial_width))    
        serial_cmaps.append(padToFixedHeightWidth(cmap,cell_height,serial_width))    
        serial_wmaps.append(padToFixedHeightWidth(wmap,cell_height,serial_width))    
    


    #--------------------------------------------
    # table_mask
    #--------------------------------------------
    table_mask=createTable(len(prod_images)+1,len(header_images)+1,2,[img.shape[1] for img in header_images],cell_height)
    
    regions,region=tableTextRegions(table_mask,[img.shape[1] for img in header_images])
    
    # dilate table
    table_mask=255-table_mask
    ksize=random.randint(3,10)
    kernel = np.ones((ksize,ksize), np.uint8)
    table_mask= cv2.dilate(table_mask, kernel, iterations=1)
    table_mask=255-table_mask
    
    # region fillup 
    printed=np.zeros(table_mask.shape)
    printed_cmap=np.zeros(table_mask.shape)
    printed_wmap=np.zeros(table_mask.shape)
    # header regs
    #{"serial":slt_serial, "brand":slt_brand,"total":slt_total,"others":slt_others}
    header_regions=[regions["serial"][0]]+[regions["brand"][0]]+regions["others"]
    for reg_val,word,cmap,wmap in zip(header_regions,header_images,header_cmaps,header_wmaps):
        printed+=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap+=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap+=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        region[region==reg_val]=0

    # total fillup
    printed+=placeWordOnMask(total_img,region,regions["total"][0],printed,fill=True)
    printed_cmap+=placeWordOnMask(total_cmap,region,regions["total"][0],printed_cmap,fill=True)
    printed_wmap+=placeWordOnMask(total_wmap,region,regions["total"][0],printed_wmap,fill=True)
    
    region[region==regions["total"][0]]=0
    # product fillup
    product_regions=regions["brand"][1:]
    for reg_val,word,cmap,wmap in zip(product_regions,prod_images,prod_cmaps,prod_wmaps):
        printed+=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap+=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap+=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        
        region[region==reg_val]=0
    
    # serial fillup
    serial_regions=regions["serial"][1:]
    for reg_val,word,cmap,wmap in zip(serial_regions,serial_images,serial_cmaps,serial_wmaps):
        printed+=placeWordOnMask(word,region,reg_val,printed,fill=True)
        printed_cmap+=placeWordOnMask(cmap,region,reg_val,printed_cmap,fill=True)
        printed_wmap+=placeWordOnMask(wmap,region,reg_val,printed_wmap,fill=True)
        
        region[region==reg_val]=0

    #--------------------------------------------
    # image construction
    #--------------------------------------------    
    img=np.copy(printed)
    table_mask[table_mask>0]=1
    table_mask=1-table_mask
    img=img+table_mask
    img[img>0]=1
    #--------------------------------------------
    # place handwritten words
    #--------------------------------------------
    ## place table
    region_values=sorted(np.unique(region))[1:]
    region_values=[int(v) for v in region_values]
    max_regs=len(region_values)
    if max_regs<place.table_min:
        place.table_min=max_regs
    len_regs=random.randint(place.table_min,place.table_min*2)
    
    hw=np.zeros_like(region)
    table_cmap=np.zeros_like(region)
    table_wmap=np.zeros_like(region)
    for i in range(len_regs):
        reg_val=random.choice(region_values)
        region_values.remove(reg_val)
        df=random.choice([n_df,nsdf])
        comps=rand_hw_word(df,place.min_num_len,place.max_num_len)
        word,cmap,wmap=createHandwritenWords(df,comps,PAD,place.comp_dim)
        
        if random.choices(population=[1,0],weights=place.rot_weights,k=1)[0]==1:
            angle=random.randint(place.min_rot,place.max_rot)
            angle=random.choice([angle,-1*angle])
            word=rotate_image(word,angle)
            wmap=rotate_image(wmap,angle)
            cmap=rotate_image(cmap,angle)
            
        ext=random.randint(0,30)
        ext_reg=random.choice([True,False])
        # words
        hw+=placeWordOnMask(word,region,reg_val,hw,ext_reg=ext_reg,fill=True,ext=ext)
        table_cmap+=placeWordOnMask(cmap,region,reg_val,table_cmap,ext_reg=ext_reg,fill=True,ext=ext)
        table_wmap+=placeWordOnMask(wmap,region,reg_val,table_wmap,ext_reg=ext_reg,fill=True,ext=ext)
    
    # returning
    img=img+hw
    img[img>0]=255
    cmap=printed_cmap+table_cmap
    wmap=printed_wmap+table_wmap  
    
    img=padAllAround(img,table.pad_dim)
    cmap=padAllAround(cmap,table.pad_dim)
    wmap=padAllAround(wmap,table.pad_dim)

    h,w=img.shape
    rgb=np.ones((h,w,3))*255
    rgb[img==255]=(0,0,0)
    
    # add noise
    num_noise  =  random.randint(1,5)
    for _ in range(num_noise):
        reg_val=random.choice(region_values)
        rgb=draw_random_noise(region,reg_val,rgb)

    return rgb,cmap,wmap
