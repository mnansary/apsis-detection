#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
#---------------------------------------------------------------
def gaussian_heatmap(size=512, distanceRatio=2):
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
#----------------------------------------------------------------
def visualize_heatmap_boxes(img,heat_map,link_map):
    _, text_score = cv2.threshold(heat_map,
                                    thresh=0.4,
                                    maxval=1,
                                    type=cv2.THRESH_BINARY)
    _, link_score = cv2.threshold(link_map,
                                thresh=0.4,
                                maxval=1,
                                type=cv2.THRESH_BINARY)
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(text_score + link_score, 0, 1).astype('uint8'),
                                                                      connectivity=4)
    plt.imshow(labels)
    plt.show()
    boxes = []
    img_h,img_w=heat_map.shape
    for component_id in range(1, n_components):
        # Filter by size
        size = stats[component_id, cv2.CC_STAT_AREA]

        if size < 10:
            continue

        # If the maximum value within this connected component is less than
        # text threshold, we skip it.
        if np.max(heat_map[labels == component_id]) < 0.7:
            continue

        # Make segmentation map. It is 255 where we find text, 0 otherwise.
        segmap = np.zeros_like(heat_map)
        segmap[labels == component_id] = 255
        segmap[np.logical_and(link_score, text_score)] = 0
        x, y, w, h = [
            stats[component_id, key] for key in
            [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
        ]

        # Expand the elements of the segmentation map
        niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, sy = max(x - niter, 0), max(y - niter, 0)
        ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
        segmap[sy:ey, sx:ex] = cv2.dilate(
            segmap[sy:ey, sx:ex],
            cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))
        # idx 
        idx = np.where(segmap>0)            
        y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0])+1, np.min(idx[1]), np.max(idx[1])+1
        boxes.append([x_min,y_min,x_max,y_max])
        
    for box in boxes:
        x_min,y_min,x_max,y_max=box
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,255,0),2)
    plt.imshow(img)
    plt.show()
#----------------------------------------------------------------------------
def get_maps(cbox,gaussian_heatmap,heat_map,link_map,prev,idx):
    '''
        creates heat_map and link_map:
        args:
            cbox             : charecter bbox[ cxmin,cymin,cxmax,cymax]
            gaussian_heatmap : the original heatmap to fit
            heat_map         : image charecter heatmap
            link_map         : link_map of the word
            prev             : list of list of previous charecter center lines
            idx              : index of current charecter
    '''
    src = np.array([[0, 0], 
                    [gaussian_heatmap.shape[1], 0], 
                    [gaussian_heatmap.shape[1],gaussian_heatmap.shape[0]],
                    [0,gaussian_heatmap.shape[0]]]).astype('float32')

    
    #--------------------
    # heat map
    #-------------------
    cxmin,cymin,cxmax,cymax=cbox
    # char points
    cx1,cx2,cx3,cx4=cxmin,cxmax,cxmax,cxmin
    cy1,cy2,cy3,cy4=cymax,cymax,cymin,cymin
    heat_points = np.array([[cx1,cy1], 
                            [cx2,cy2], 
                            [cx3,cy3], 
                            [cx4,cy4]]).astype('float32')
    M_heat = cv2.getPerspectiveTransform(src=src,dst=heat_points)
    heat_map+=cv2.warpPerspective(gaussian_heatmap,M_heat, dsize=(heat_map.shape[1],heat_map.shape[0])).astype('float32')

    #-------------------------------
    # link map
    #-------------------------------
    lx2=cx1+(cx2-cx1)/2
    lx3=lx2
    y_shift=(cy4-cy1)/4
    ly2=cy1+y_shift
    ly3=cy4-y_shift
    if prev is not None:
        prev[idx]=[lx2,lx3,ly2,ly3]
        if idx>0:
            lx1,lx4,ly1,ly4=prev[idx-1]
            link_points = np.array([[lx1,ly1], [lx2,ly2], [lx3,ly3], [lx4,ly4]]).astype('float32')
            M_link = cv2.getPerspectiveTransform(src=src,dst=link_points)
            link_map+=cv2.warpPerspective(gaussian_heatmap,M_link, dsize=(link_map.shape[1],link_map.shape[0])).astype('float32')

    return heat_map,link_map,prev
