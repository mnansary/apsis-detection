# -*-coding: utf-8 -
'''
    @author: MD.Nazmuddoha Ansary, MD. Rezwanul Haque
'''
#----------------------------
# imports
#----------------------------
import numpy as np
import random
import os
import cv2
import string
import matplotlib.pyplot as plt
import pandas as pd 
from glob import glob

import PIL.Image,PIL.ImageDraw,PIL.ImageFont
from numpy.core.fromnumeric import prod

from .memo import Head,Table,Bottom,LineSection,LineWithExtension,Placement,PAD
from .memo import rand_head,rand_products,rand_word,rand_bottom,rand_hw_word

from .word import createPrintedLine,handleExtensions,createHandwritenWords
from .table import createTable,tableTextRegions

from .utils import padToFixedHeightWidth,padAllAround,placeWordOnMask,rotate_image,draw_random_noise
#----------------------------
# render capacity: toolset
#----------------------------
def renderFontMaps(LineSection,font_path):
    '''
        renders a font map
    '''
    maps={}
    sizes=LineSection.font_sizes_big+LineSection.font_sizes_mid
    for size in sizes:
        maps[str(size)]=PIL.ImageFont.truetype(font_path, size=size)
    return maps