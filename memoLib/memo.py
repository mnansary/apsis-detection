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
    no_pad_dim      =   64
    single_pad_dim  =   84
    double_pad_dim  =   104
    top             =  ['ই', 'ঈ', 'উ', 'ঊ', 'ঐ','ঔ','ট', 'ঠ',' ি', 'ী', 'ৈ', 'ৌ','ঁ','র্']
    bot             =  ['ু', 'ূ', 'ৃ',]
            
class LineSection:
    word_len_max =   7
    word_len_min =   1
    num_word_max =   5
    num_word_min =   2
    symbols      =   [".","-","/",",","।","(",")"]
    vocabs       =   ["mixed","number","grapheme"]
    vweights     =   [0.1,0.1,0.8]
    max_syms     =   1
    font_sizes_big   =   [64,48]
    font_sizes_mid   =   [32,28,24]
                                       

class LineWithExtension(LineSection):
    num_word_max =   2
    ext_types    =   ["single","double"]
    ext_symbols  =   [".","_"]

#--------------------
# format classes
#--------------------
class Table(LineSection):
    serial           ={"bn":[['সি', 'রি', 'য়া', 'ল', ':'],
                            ['ক্র', 'ম', ':'],
                            ['ক্র', 'মি', 'ক'],
                            ['ন', 'ম্ব', 'র'],
                            ['ন', 'ং']],
                        "en":[['s', 'e', 'r', 'i', 'a', 'l'], 
                              ['s', ':'],
                              ['n', 'u', 'm', ':'], 
                              ['n', ':']]}
    num_product_min  =  5
    num_product_max  =  25
    num_extCOL_min   =  3
    num_extCOL_max   =  6
    pad_dim          =  10
    
    vweights         =  [0.05,0.05,0.9]
    products         =  []
    column_names     =  []
     

class Head:
    min_line_section    =3
    max_line_section    =5
    min_single_exts     =2
    max_single_exts     =5
    min_double_exts     =1
    max_double_exts     =3

    line_sections       =[]   # [{words,font_size}]
    single_exts         =[]   # [{words,font_size,ext,ext_len}]
    double_exts         =[]   # [{words,font_size,ext,ext_len},{words,font_size,ext,ext_len}]

    
#--------------------
# text-functions
#--------------------
def rand_word(vocab,symbol,max_len,min_len,add_space=True):
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
    if add_space:
        comps.append(" ")
    return comps


def rand_line_section(section,graphemes,numbers):
    '''
        creates a random line with given properties and sections
    '''
    line=[]
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
        line.append(word)
    return line 


def rand_line_extension(section,graphemes,numbers,ext_type):
    '''
        creates a random line with given properties and sections
    '''
    # single or double
    if ext_type=="single":
        line=rand_line_section(section,graphemes,numbers)
        return line
    else:
        line1=rand_line_section(section,graphemes,numbers)
        line2=rand_line_section(section,graphemes,numbers)
        return line1,line2
#--------------------
# head-functions
#--------------------


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
        head.line_sections.append([{"words":rand_line_section(line_section,graphemes,numbers),
                                   "font_size":random.choice(line_section.font_sizes_big)}])
    
    
    # add double ext sections
    num_double_sections=random.randint(head.min_double_exts,head.max_double_exts)
    for _ in range(num_double_sections):
        line1,line2=rand_line_extension(line_ext,graphemes,numbers,"double")
        font_size=random.choice(line_ext.font_sizes_mid)
        data=[{"words":line1,"font_size":font_size},
              {"words":line2,"font_size":font_size}]

        head.double_exts.append(data)
    

    # add single ext sections
    num_double_sections=random.randint(head.min_single_exts,head.max_single_exts)
    for _ in range(num_double_sections):
        line=rand_line_extension(line_ext,graphemes,numbers,"single")
        data={"words":line}
        data["font_size"]=random.choice(line_ext.font_sizes_mid)
        head.single_exts.append([data])
    return head


#--------------------
# table-functions
#--------------------


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
        table.products.append([{"words":rand_line_section(table,graphemes,numbers),"font_size":table.font_size}])
    
    return table

if __name__=="__main__":
    graphemes =  list(string.ascii_lowercase)
    numbers   =  [str(i) for i in range(10)]
    #rand_table(graphemes,numbers,Table)    
