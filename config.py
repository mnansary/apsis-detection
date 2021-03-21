# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
import os 
from glob import glob 
#--------------------
# config for data gen
#--------------------
class CONFIG:
    # number of lines per image
    MIN_NUM_LINES   =   1
    MAX_NUM_MINES   =   30
    # number of words per line
    MIN_NUM_WORDS   =   1
    MAX_NUM_WORDS   =   10
    # using numbers
    USE_NUMS        =   True
    # xy plane rotation   
    ROTATE_XY_MIN   =   0
    ROTATE_XY_MAX   =   30
    # perspective trsnsformation
    ROTATE_Z_MIN    =   0
    ROTATE_Z_MAX    =   30
    # word level rotation
    ROTATE_WORD_MIN =   0
    ROTATE_WORD_MAX =   30
    # use date separators to create synthetic dates
    DATE_SEP        =   True
    # csv paths for datset core elements
    GRAPHEME_CSV    =   os.path.join(os.getcwd(),"resources","graphemes.csv")
    NUMBER_CSV      =   os.path.join(os.getcwd(),"resources","numbers.csv")   
    # separator paths
    SEPARATORS      =   [_path for _path in glob(os.path.join(os.getcwd(),"resources","separators","*.*"))]
    # number of dates per page (date format)
    NUM_DATES       =   1 