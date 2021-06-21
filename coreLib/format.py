# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import cv2
import numpy as np
#--------------------
# format
#--------------------

def convertToTotalText(page,labels,orientation = 'h'):
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
    # your code starts from here 
    # after finalization change returns segment under doc string above
    
    # char mask
    char_mask=np.zeros(page.shape)
    for line_labels in labels:
        for label in line_labels:
            for k,v in label.items():
                if v!=' ':
                    char_mask[page==k]=255

    char_mask=np.expand_dims(char_mask,axis=-1)
    char_mask=np.concatenate([char_mask,char_mask,char_mask],axis=-1)
    char_mask=char_mask.astype("uint8")
    
    
    
    # text lines
    text_lines=[]
    # word_mask
    word_mask=np.zeros(page.shape)
    
    
    for line_labels in labels:
        for label in line_labels:
            _label_mask=np.zeros((page.shape), dtype=np.uint8)
            transcriptions=''
            for k,v in label.items():
                if v!=' ':
                    transcriptions+=v
                    idx = np.where(page==k)
                    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                    word_mask[y_min:y_max+1,x_min:x_max+1]=255
                    _label_mask[y_min:y_max+1,x_min:x_max+1]=255
            
            
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
        
            
            coordinates = "x: " + str(np.array(X)) + ", " + "y: " + str(np.array(Y)) + ", "+"ornt: "+ str([orientation]) + ", "+ "transcriptions: " + str([transcriptions])  
            text_lines.append(coordinates)
            
    word_mask=np.expand_dims(word_mask,axis=-1)
    word_mask=np.concatenate([word_mask,word_mask,word_mask],axis=-1)
    word_mask=word_mask.astype("uint8")
    
    return char_mask,word_mask,text_lines