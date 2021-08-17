# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary, MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
from .render import *

#----------------------------
# render capacity: bottom 
#----------------------------
def renderMemoBottom(ds,language,max_width,pad_dim=10):
    """
        @function author:        
        Create image of table part of Memo
        args:
            ds         = dataset object that holds all the paths and resources
            language   = a specific language to use
            iden       = a specific identifier for marking    
        
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
    noise_signs =  [img_path for img_path in glob(os.path.join(ds.common.noise.sign,"*.bmp"))]
    #--------------------------------------------
    # text gen section
    #--------------------------------------------
    bottom=Bottom()
    maps=renderFontMaps(bottom,random.choice(font_paths))
    # fill-up texts
    bottom=rand_bottom(graphemes,numbers,bottom)
    ## image
    h_max=0
    w_max=0
    
    #--------------------------------------------
    # create line sections
    #--------------------------------------------
    # sender
    data=bottom.sender_line[0]
    sender_img,sender_cmap,sender_wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
    h,w=sender_img.shape
    if h>h_max:h_max=h
    if w>w_max:w_max=w
    # receiver
    data=bottom.reciver_line[0]
    rec_img,rec_cmap,rec_wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
    h,w=rec_img.shape
    if h>h_max:h_max=h
    if w>w_max:w_max=w
    # signing
    sign_images=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in [sender_img,rec_img]]
    sign_cmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in [sender_cmap,rec_cmap]]
    sign_wmaps=[padToFixedHeightWidth(line_img,h_max,w_max) for line_img in [sender_wmap,rec_wmap]]
    # middle
    data=bottom.middle_line[0]
    mid_img,mid_cmap,mid_wmap=createPrintedLine(text=data["line"],font=maps[str(data["font_size"])])
    height=mid_img.shape[0]+2*pad_dim
    mid_img=padToFixedHeightWidth(mid_img,height,max_width)
    mid_cmap=padToFixedHeightWidth(mid_cmap,height,max_width)
    mid_wmap=padToFixedHeightWidth(mid_wmap,height,max_width)
    # pad 
    mid_pad=np.zeros((h_max,max_width//2))

    sign_images=[sign_images[0]*3,mid_pad,sign_images[-1]*4]
    sign_wmaps =[sign_wmaps[0],mid_pad,sign_wmaps[-1]]
    sign_cmaps =[sign_cmaps[0],mid_pad,sign_cmaps[-1]]
    
    sign_img=np.concatenate(sign_images,axis=1)
    sign_wmap=np.concatenate(sign_wmaps,axis=1)
    sign_cmap=np.concatenate(sign_cmaps,axis=1)

    h,w=sign_img.shape
    height=sign_img.shape[0]+2*pad_dim
    sign_img=padToFixedHeightWidth(sign_img,height,max_width)
    sign_cmap=padToFixedHeightWidth(sign_cmap,height,max_width)
    sign_wmap=padToFixedHeightWidth(sign_wmap,height,max_width)
    
    # print_mask
    if random.choice([0,1])==1:
        printed=np.concatenate([sign_img,mid_img],axis=0)
        cmap=np.concatenate([sign_cmap,mid_cmap],axis=0)
        wmap=np.concatenate([sign_wmap,mid_wmap],axis=0)
    else:
        printed=np.concatenate([mid_img,sign_img],axis=0)
        cmap=np.concatenate([mid_cmap,sign_cmap],axis=0)
        wmap=np.concatenate([mid_wmap,sign_wmap],axis=0)
    # image mask
    img=np.copy(printed)
    region=np.zeros_like(img)
    rid=5
    #######################
    # fixed region
    for i in [3,4]:
        idx=np.where(img==i)
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])            
        region[y_min:y_max,x_min:x_max]=rid
        rid+=1
    #######################
    
    # place bottom
    noise_num=random.choice([1,2])
    region_values=sorted(np.unique(region))[1:]
    hw=np.zeros_like(region)
    for i in range(noise_num):
        word=cv2.imread(random.choice(noise_signs),0)
        word=255-word
        word[word>0]=1
        reg_val=region_values[i]
        hw+=placeWordOnMask(word,region,reg_val,hw,ext_reg=True,fill=False,ext=(10,30))
    img+=hw
    img[img>0]=255
    h,w=img.shape
    h_new=int(max_width*h/w)
    img = cv2.resize(img, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    cmap = cv2.resize(cmap, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    wmap = cv2.resize(wmap, (max_width,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    rgb=np.ones((h,w,3))*255
    rgb[img==255]=(0,0,0)
    return rgb,cmap,wmap