# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary, MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
from .render import *

#----------------------------
# render capacity: memo head
#----------------------------
def renderMemoHead(ds,language,max_width):


    """
        @function author:        
        Create image of top part of Memo
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
    # text gen section
    #--------------------------------------------
    head=Head()
    lineSection=LineSection()
    lineWithExtension=LineWithExtension()
    place=Placement()
    head=rand_head(graphemes,numbers,head,lineSection,lineWithExtension)
    maps=renderFontMaps(lineSection,random.choice(font_paths))
    ext_sym=random.choice(lineWithExtension.ext_symbols)
    #--------------------------------------------
    # image gen section
    #--------------------------------------------
    reg_iden=5
    h_max=0
    w_max=max_width
    line_images=[]
    line_wmaps =[]
    line_cmaps =[]
    #--------------------------------------------
    # create line sections
    #--------------------------------------------
    for line_data in head.line_sections:
        assert len(line_data)==1
        data=line_data[0]
        img,cmap,wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
        h,w=img.shape
        if h>h_max:h_max=h
        if w>w_max:w_max=w
        # append
        assert img.shape==cmap.shape==wmap.shape
        line_images.append(img)
        line_cmaps.append(cmap)
        line_wmaps.append(wmap)
        
        
    line_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in line_images]
    line_cmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in line_cmaps]
    line_wmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in line_wmaps]
    #--------------------------------------------
    # create double ext sections
    #--------------------------------------------
    for data in head.double_exts:
        LINE1=True
        LINE2=True
        assert len(data)==2
        img1,cmap1,wmap1=createPrintedLine(text=data[0]["line"],font=maps[str(data[0]["font_size"])])
        # add ext
        h1,w1=img1.shape
        ext_w=w_max//2-w1
        if ext_w>0:
            ext=np.ones((h1,ext_w))*reg_iden
            reg_iden+=1        
            img1=np.concatenate([img1,ext],axis=1)
            cmap1=np.concatenate([cmap1,np.zeros_like(ext)],axis=1)
            wmap1=np.concatenate([wmap1,np.zeros_like(ext)],axis=1)
            
        else:
            LINE1=False
            img1=np.zeros((h1,w_max//2))
            cmap1=np.zeros((h1,w_max//2))
            wmap1=np.zeros((h1,w_max//2))

        img2,cmap2,wmap2=createPrintedLine(text=data[0]["line"],font=maps[str(data[0]["font_size"])])
        # add ext
        h2,w2=img2.shape
        if ext_w>0:    
            ext_w=w_max//2-w2
            ext=np.ones((h2,ext_w))*reg_iden 
            reg_iden+=1
            img2=np.concatenate([img2,ext],axis=1)
            cmap2=np.concatenate([cmap2,np.zeros_like(ext)],axis=1)
            wmap2=np.concatenate([wmap2,np.zeros_like(ext)],axis=1)
            
        else:
            LINE2=False
            img2=np.zeros((h2,w_max//2))
            cmap2=np.zeros((h2,w_max//2))
            wmap2=np.zeros((h2,w_max//2))
        
        img=np.concatenate([img1,img2],axis=1)
        cmap=np.concatenate([cmap1,cmap2],axis=1)
        wmap=np.concatenate([wmap1,wmap2],axis=1)
        
        # correction
        h,w=img.shape
        if w<w_max:
            pad=np.zeros((h,w_max-w))
            img=np.concatenate([img,pad],axis=1)
            cmap=np.concatenate([cmap,pad],axis=1)
            wmap=np.concatenate([wmap,pad],axis=1)
            
        # append
        assert img.shape==cmap.shape==wmap.shape
        line_images.append(img)
        line_cmaps.append(cmap)
        line_wmaps.append(wmap)
        
    #--------------------------------------------
    # create single ext sections
    #--------------------------------------------
    for line_data in head.single_exts:
        assert len(line_data)==1
        data=line_data[0]
        img,cmap,wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
        # add ext
        h,w=img.shape
        ext_w=w_max-w
        if ext_w>0:
            ext=np.ones((h,ext_w))*reg_iden
            reg_iden+=1
            img=np.concatenate([img,ext],axis=1)
            cmap=np.concatenate([cmap,np.zeros_like(ext)],axis=1)
            wmap=np.concatenate([wmap,np.zeros_like(ext)],axis=1)
            
        # append
        assert img.shape==cmap.shape==wmap.shape
        line_images.append(img)
        line_cmaps.append(cmap)
        line_wmaps.append(wmap)
        
    #-----------------------------
    # format masks
    #-----------------------------
    img=np.concatenate(line_images,axis=0)
    
    cmap=np.concatenate(line_cmaps,axis=0)
    wmap=np.concatenate(line_wmaps,axis=0)
    printed=np.zeros_like(img)
    region =np.zeros_like(img)
    printed[img==1]=1
    #---------------------------------
    # fix image
    #--------------------------------
    ext_data=np.zeros_like(img)
    for v in sorted(np.unique(img))[2:]:
        # ext_image
        idx=np.where(img==v)
        if len(idx[0])>0:
            x_min,x_max = np.min(idx[1]), np.max(idx[1])
            width=x_max-x_min
            if width > max_width//4:
                region[img==v]=v
            ext_word=handleExtensions(ext_sym,maps[str(lineSection.font_sizes_mid[-1])],width)
            if ext_word is not None:
                # place
                ext_data+=placeWordOnMask(ext_word,img,v,ext_data,fill=True)
                img[img==v]=0
                
    img=img+ext_data
    img[img>0]=255
            
    #---------------------------------
    # place handwritten 
    #---------------------------------
    region_values=sorted(np.unique(region))[1:]
    region_values=[int(v) for v in region_values]
    hw=np.zeros_like(img)
    if len(region_values)>0:
        max_regs=len(region_values)
        if max_regs<place.head_min:
            place.head_min=max_regs
        len_regs=random.randint(place.head_min,max_regs)
        
        for i in range(len_regs):
            reg_val=random.choice(region_values)
            region_values.remove(reg_val)
            df=random.choice([g_df,gsdf,adf,nsdf])
            comps=rand_hw_word(df,place.min_word_len,place.max_word_len)
            word,wcmap,wwmap=createHandwritenWords(df,comps,PAD,place.comp_dim)
            # words
            ext=random.randint(10,30)
            hw+=placeWordOnMask(word,region,reg_val,hw,ext_reg=True,fill=False,ext=ext)
            cmap+=placeWordOnMask(wcmap,region,reg_val,cmap,ext_reg=True,fill=False,ext=ext)
            wmap+=placeWordOnMask(wwmap,region,reg_val,wmap,ext_reg=True,fill=False,ext=ext)
        
    #-----------------------------------
    # image form
    #-----------------------------------
    img=img+hw
    img[img>0]=255
    h,w=img.shape
    h_new=int(max_width*h/w)
    img = cv2.resize(img, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    cmap = cv2.resize(cmap, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    wmap = cv2.resize(wmap, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    rgb=np.ones((h,w,3))*255
    rgb[img==255]=(0,0,0)

    return rgb,cmap,wmap
