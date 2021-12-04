# -*-coding: utf-8 -
'''
    @author: md. nazmuddoha ansary
'''

class config:
    # number of lines per image
    min_num_lines   =   1
    max_num_lines   =   10
    # number of words per line
    min_num_words   =   1
    max_num_words   =   10
    # word lenght
    min_word_len    =   1
    max_word_len    =   10
    # num lenght
    min_num_len     =   1
    max_num_len     =   10
    # comp dimension
    comp_dim        =   64
    
    # word space
    word_min_space  =   50
    word_max_space  =   100
    
    vert_min_space  =   1
    vert_max_space  =   100

    back_dim        =   1024
    
    heatmap_ratio   =  2
    max_warp_perc   =  20

    class data:
        sources     =   ["bangla","english"]
        formats     =   ["handwritten","printed"]
        components  =   ["number","grapheme","mixed"]
