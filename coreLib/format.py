# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary,MD. Rezwanul Haque
'''
#--------------------
# imports
#--------------------
import cv2
import numpy as np
#--------------------
# format
#--------------------

def TotalText(page,labels):
    '''
        **_label_mask:polygon
        ** text_lines[0]=
            "x: [[214 280 362 349 284 231]], y: [[325 290 320 347 316 346]], ornt: [u'c'], transcriptions: [u'ASRAMA']"
        @author
        create a function to convert page image to total text format data
        This should not depend on- 
            * language or 
            * type (handwritten/printed) or 
            * data(number/word/symbol)
        args:
            page   :     marked image of a page given at letter by letter 
            labels :     list of markings for each word
        returns:
            whatever is necessary for the total-text format
        FUTURE:
            * Rotation will be added after render class 
    '''
    # text lines
    text_lines=[]
    # word_mask
    word_mask=np.zeros(page.shape)
    # char mask
    char_mask=np.zeros(page.shape)
    
    for line_labels in labels:
        for label in line_labels:
            _label_mask=np.zeros((page.shape), dtype=np.uint8)
            transcriptions=''
            _ymins=[]
            _ymaxs=[]
            _xmins=[]
            _xmaxs=[] 
            
            
            for k,v in label.items():
                if v!=' ':
                    char_mask[page==k]=255
                    
                    transcriptions+=v
                    idx = np.where(page==k)
                    
                    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                    _ymins.append(y_min)
                    _ymaxs.append(y_max)
                    _xmins.append(x_min)
                    _xmaxs.append(x_max)            
                    
            
            
            _label_mask[min(_ymins):max(_ymaxs)+1,min(_xmins):max(_xmaxs)+1]=255
            word_mask[min(_ymins):max(_ymaxs)+1,min(_xmins):max(_xmaxs)+1]=255
            
            contours, hiearchy = cv2.findContours(_label_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            X = []
            Y = []
            x_cord=[]
            y_cord=[]
            for i in range(len(contours)):
                for j in range(len(contours[i])):
                    x_cord.append(contours[i][j][0][0])
                    y_cord.append(contours[i][j][0][1])
            
            X.append(x_cord)
            Y.append(y_cord)
            orientation="u'h'"
            coordinates = "x: " + str(np.array(X)) + ", " + "y: " + str(np.array(Y)) + ", "+"ornt: "+ "["+orientation +"]"+", "+ "transcriptions: " + f"[u'{transcriptions}']"  
            text_lines.append(coordinates)
            
            
    char_mask=np.expand_dims(char_mask,axis=-1)
    char_mask=np.concatenate([char_mask,char_mask,char_mask],axis=-1)
    char_mask=char_mask.astype("uint8")
    
    word_mask=np.expand_dims(word_mask,axis=-1)
    word_mask=np.concatenate([word_mask,word_mask,word_mask],axis=-1)
    word_mask=word_mask.astype("uint8")
    
    return char_mask,word_mask,text_lines


#--------------------
# linetext-format
#--------------------
def lineText(page,labels,heatmap):
    '''
        @author
        args:
            page   :     marked image of a page given at letter by letter 
            labels :     list of markings for each word
        returns:
            charmap,wordmap,heatmap
         
    '''
    # source bbobx of heatmap
    src = np.array([[0, 0], 
                    [heatmap.shape[1], 0], 
                    [heatmap.shape[1], heatmap.shape[0]],
                    [0, heatmap.shape[0]]]).astype('float32')

    # word_mask
    word_mask=np.zeros(page.shape)
    # char mask
    char_mask=np.zeros(page.shape)
    # heat mask
    heat_mask=np.zeros(page.shape)
    
    
    for line_labels in labels:
        
        for label in line_labels:
            _ymins=[]
            _ymaxs=[]
            _xmins=[]
            _xmaxs=[] 

            for k,v in label.items():
                if v!=' ':
                    # char mask
                    char_mask[page==k]=255
                    idx = np.where(page==k)
                    
                    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                    _ymins.append(y_min)
                    _ymaxs.append(y_max)
                    _xmins.append(x_min)
                    _xmaxs.append(x_max)            
                    # heat mask    
                    x1=x_min
                    y1=y_max
                    x2=x_max
                    y2=y_max
                    x3=x_max
                    y3=y_min
                    x4=x_min
                    y4=y_min
                    _points = np.array([[x1, y1], 
                                        [x2, y2], 
                                        [x3, y3], 
                                        [x4, y4]]).astype('float32') 
                    # transforms the bbox and creates the heatmap
                    M = cv2.getPerspectiveTransform(src=src,dst=_points)
                    heat_mask+= cv2.warpPerspective(heatmap,
                                                    M, 
                                                    dsize=(heat_mask.shape[1],
                                                           heat_mask.shape[0])).astype('float32')

        
        # word mask
        x_min=min(_xmins)
        x_max=max(_xmaxs)
        y_min=min(_ymins)
        y_max=max(_ymaxs)

        x1=x_min
        y1=y_max
        x2=x_max
        y2=y_max
        x3=x_max
        y3=y_min
        x4=x_min
        y4=y_min
        word_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype('float32') 
        # transforms the bbox and creates the heatmap
        M = cv2.getPerspectiveTransform(src=src,dst=word_points)
        word_mask+= cv2.warpPerspective(heatmap,
                                        M, 
                                        dsize=(word_mask.shape[1],
                                               word_mask.shape[0])).astype('float32')

    
    char_mask=char_mask.astype("uint8")
    word_mask=word_mask.astype("uint8")
    heat_mask=heat_mask.astype("uint8")
    return char_mask,word_mask,heat_mask
