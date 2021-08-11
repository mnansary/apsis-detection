# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import string
#--------------------
# base classes
#--------------------
class PAD:
    no_pad_dim      =   (64,64)
    single_pad_dim  =   (84,84)
    double_pad_dim  =   (104,104)
    top             =  ['ই', 'ঈ', 'উ', 'ঊ', 'ঐ','ঔ','ট', 'ঠ',' ি', 'ী', 'ৈ', 'ৌ','ঁ','র্']
    bot             =  ['ু', 'ূ', 'ৃ',]
    height          =   20 
            
class LineSection(object):
    def __init__(self):
        self.word_len_max =   7
        self.word_len_min =   3
        self.num_word_max =   4
        self.num_word_min =   2
        self.symbols      =   [".","-","/",",","।","(",")"]
        self.vocabs       =   ["mixed","number","grapheme"]
        self.vweights     =   [0.1,0.1,0.8]
        self.max_syms     =   1
        self.font_sizes_big   =   [128,112,96]
        self.font_sizes_mid   =   [80,64]
                                        

class LineWithExtension(LineSection):
    def __init__(self):
        super().__init__()
        self.num_word_max =   2
        self.ext_types    =   ["single","double"]
        self.ext_symbols  =   [".","_"]

#--------------------
# format classes
#--------------------

        

#--------------------
# text-functions
#--------------------
def rand_word(vocab,symbol,max_len,min_len):
    '''
        creates a random word
        args:
            vocab    : list of graphemes
            symbol   : symbol to add
            max_len  : maximum length of word
            min_len  : min length of word
    '''
    comps=[]
    len_word=random.randint(min_len,max_len)
    for i in range(len_word):
        comps.append(random.choice(vocab))
    if symbol is not None:
        comps.append(symbol)
    comps.append(" ")
    word="".join(comps)
    return word


def rand_line(section,graphemes,numbers):
    '''
        creates a random line with given properties and sections
    '''
    line=''
    sym_count=0
    max_sym  =random.randint(0,section.max_syms)
    num_word=random.randint(section.num_word_min,section.num_word_max)
    for i in range(num_word):
        _vocab=random.choices(population=section.vocabs,weights=section.vweights,k=1)[0]
        if _vocab=="mixed":
            vocab=graphemes+numbers
        elif _vocab=="grapheme":
            vocab=graphemes
        else:
            vocab=numbers

        if random.choices(population=[1,0],weights=[0.1, 0.9],k=1)[0]==1:
            if sym_count<=max_sym:
                symbol=random.choice(section.symbols)
            else:
                symbol=None
        else:
            symbol=None
        
        word=rand_word(vocab,symbol,section.word_len_max,section.word_len_min)
        line+=word
    return line 


def rand_line_with_extension(section,graphemes,numbers,ext_type):
    '''
        creates a random line with given properties and sections
    '''
    # single or double
    if ext_type=="single":
        line=rand_line(section,graphemes,numbers)
        return line
    else:
        line1=rand_line(section,graphemes,numbers)
        line2=rand_line(section,graphemes,numbers)
        return line1,line2
#--------------------
# head-functions
#--------------------
class Head(object):
    def __init__(self):
        
        self.min_line_section    =2
        self.max_line_section    =4
        self.min_single_exts     =1
        self.max_single_exts     =2
        self.min_double_exts     =1
        self.max_double_exts     =2

        self.line_sections       =[]   # [{words,font_size}]
        self.single_exts         =[]   # [{words,font_size,ext,ext_len}]
        self.double_exts         =[]   # [{words,font_size,ext,ext_len},{words,font_size,ext,ext_len}]


def rand_head(graphemes,numbers,head,line_section,line_ext):
    '''
        generates random head data
        args:
            graphemes   :   list of valid graphemes to use
            numbers     :   list of valid number to use
            head        :   head class
            line_section:   line_section class
            line_ext    :   line extension class
    '''
    # add line sections
    num_line_sections=random.randint(head.min_line_section,head.max_line_section)
    for _ in range(num_line_sections):
        head.line_sections.append([{"line":rand_line(line_section,graphemes,numbers),
                                    "font_size":random.choice(line_section.font_sizes_big)}])
    
    
    
    font_size=random.choice(line_ext.font_sizes_mid)
    # add double ext sections
    num_double_sections=random.randint(head.min_double_exts,head.max_double_exts)
    for _ in range(num_double_sections):
        line1,line2=rand_line_with_extension(line_ext,graphemes,numbers,"double")
        
        data=[{"line":line1,"font_size":font_size},
              {"line":line2,"font_size":font_size}]

        head.double_exts.append(data)
    

    # add single ext sections
    num_single_sections=random.randint(head.min_single_exts,head.max_single_exts)
    for _ in range(num_single_sections):
        line=rand_line_with_extension(line_ext,graphemes,numbers,"single")
        data={"line":line}
        data["font_size"]=font_size
        head.single_exts.append([data])
    return head


# #--------------------
# # table-functions
# #--------------------
class Table(LineSection):
    def __init__(self):
        super().__init__()
    
        self.serial           ={"bn":['সিরিয়াল:', 'ক্রম:', 'ক্রমিক', 'নম্বর', 'নং'],
                                "en":['serial', 's:', 'num:', 'n:']}
        self.num_product_min  =  5
        self.num_product_max  =  15
        self.num_extCOL_min   =  3
        self.num_extCOL_max   =  7
        
        self.vweights         =  [0.05,0.05,0.9]
        self.products         =  []
        self.column_names     =  []
        self.pad_dim          =  10
def rand_products(graphemes,numbers,table):
    '''
        generates random head data
        args:
            graphemes   :   list of valid graphemes to use
            numbers     :   list of valid number to use
            table       :   table class
    '''
    # add line sections
    table.font_size=random.choice(table.font_sizes_mid)
    num_line_sections=random.randint(table.num_product_min,table.num_product_max)
    for _ in range(num_line_sections):
        table.products.append([{"line":rand_line(table,graphemes,numbers),"font_size":table.font_size}])
    
    return table

#--------------------
# bottom-functions
#--------------------
class Bottom(object):
    def __init__(self):
        super().__init__()
        self.sender_line  =[]
        self.reciver_line =[]
        self.middle_line  =[]
        self.pad          =10
        self.word_len_max =   5
        self.word_len_min =   3
        self.num_word_max =   1
        self.num_word_min =   1
        self.symbols      =   [".","-","/",",","।","(",")"]
        self.vocabs       =   ["mixed","number","grapheme"]
        self.vweights     =   [0.1,0.1,0.8]
        self.max_syms     =   1
        self.font_sizes_big   =   [128,112,96]
        self.font_sizes_mid   =   [80,64]
        

def rand_bottom(graphemes,numbers,bottom):
    '''
        generates random head data
        args:
            graphemes   :   list of valid graphemes to use
            numbers     :   list of valid number to use
            bottom       :   bottom class
    '''
    # add line sections
    bottom.font_size=bottom.font_sizes_mid[0]
    bottom.sender_line.append({"line":rand_line(bottom,graphemes,numbers),"font_size":bottom.font_size})
    bottom.reciver_line.append({"line":rand_line(bottom,graphemes,numbers),"font_size":bottom.font_size})
    bottom.num_word_max=5
    bottom.num_word_max=3
    bottom.middle_line.append({"line":rand_line(bottom,graphemes,numbers),"font_size":bottom.font_size})
    
    return bottom
#--------------------
# placement-functions
#--------------------
class Placement(object):
    def __init__(self):
        self.head_min        =2
        self.min_word_len    =2
        self.max_word_len    =5 
        self.comp_dim        =64 
        self.table_min       =3 
        self.min_num_len     =2
        self.max_num_len     =4 
        self.max_rot         = 45
        self.min_rot         = 5
        self.rot_weights     = [0.3,0.7]
        self.max_noise       = 3

def rand_hw_word(df,min_word_len,max_word_len):
    '''
        comps for handwritten word
    '''
    comps=[]
    len_word=random.randint(min_word_len,max_word_len)
    for _ in range(len_word):
        idx=random.randint(0,len(df)-1)
        comps.append(df.iloc[idx,1])
    return comps  