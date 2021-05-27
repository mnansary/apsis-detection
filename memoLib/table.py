# -*-coding: utf-8 -
'''
    @author: MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np 
import cv2 
#----------------------------
# table function
#----------------------------
def createTable(num_rows,    
                num_columns, 
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