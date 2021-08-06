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
class LineSection:
    word_len_max =   10
    word_len_min =   1
    num_word_max =   5
    num_word_min =   2
    symbols      =   [".","-","/",",","।","(",")"]
    vocabs       =   ["mixed","number","grapheme"]
    max_syms     =   3
    font_sizes_big   =   [128,96,64]
    font_sizes_mid   =   [64,32,16]
                                       

class LineWithExtension(LineSection):
    num_word_max =   2
    ext_types    =   ["single","double"]
    ext_symbols  =   ["#",".","_"]

#--------------------
# format classes
#--------------------

class Head:
    min_line_section    =3
    max_line_section    =10
    min_single_exts     =2
    max_single_exts     =5
    min_double_exts     =1
    max_double_exts     =3
    max_data_len        =100

    line_sections       =[]   # {"comps":,"font_size:,inverted:"}
    single_exts         =[]   # {"comps":,"font_size:,"ext","ext_len"}
    double_exts         =[]   # {"c1","c2",font_size,"ext","l1","l2"}

    
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
        if random.choice(section.vocabs)=="mixed":
            vocab=graphemes+numbers
        elif random.choice(section.vocabs)=="word":
            vocab=graphemes
        else:
            vocab=numbers

        if random.choices(population=[0,1],weights=[0.1, 0.9],k=1)[0]==1:
            if sym_count<=max_sym:
                symbol=random.choice(section.symbols)
            else:
                symbol=None
        else:
            symbol=None
        line+=rand_word(vocab,
                        symbol,
                        section.word_len_max,
                        section.word_len_min)
    return line 


def rand_line_extension(section,graphemes,numbers,max_data,ext_type):
    '''
        creates a random line with given properties and sections
    '''
    _ext     =random.choice(section.ext_symbols)
    # single or double
    if ext_type=="single":
        line=rand_line_section(section,graphemes,numbers)
        ext_len=max_data-len(line)
        return line,_ext,ext_len
    else:
        line1=rand_line_section(section,graphemes,numbers)
        ext_len1=max_data//2-len(line1)
        line2=rand_line_section(section,graphemes,numbers)
        ext_len2=max_data//2-len(line2)
        return line1,line2,_ext,ext_len1,ext_len2
#--------------------
# memo-functions
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
        head.line_sections.append({"comps":rand_line_section(line_section,graphemes,numbers),
                                   "font_size":random.choice(line_section.font_sizes_big),
                                   "inverted":random.choices(population=[True,False],weights=[0.2, 0.8],k=1)[0]})
    
    
    # add double ext sections
    num_double_sections=random.randint(head.min_double_exts,head.max_double_exts)
    for _ in range(num_double_sections):
        c1,c2,ext,l1,l2=rand_line_extension(line_ext,graphemes,numbers,head.max_data_len,"double")
        _dict={"c1":c1,"c2":c2,"l1":l1,"l2":l2,"ext":ext}
        _dict["font_size"]=random.choice(line_ext.font_sizes_mid)
        head.double_exts.append(_dict)
    

    # add single ext sections
    num_double_sections=random.randint(head.min_single_exts,head.max_single_exts)
    for _ in range(num_double_sections):
        line,_ext,ext_len=rand_line_extension(line_ext,graphemes,numbers,head.max_data_len,"single")
        _dict={"c":line,"l":ext_len,"ext":ext}
        _dict["font_size"]=random.choice(line_ext.font_sizes_mid)
        head.single_exts.append(_dict)
    
        
    return head
