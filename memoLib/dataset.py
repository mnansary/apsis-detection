# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd 
from glob import glob
from tqdm.auto import tqdm
from ast import literal_eval
from .utils import LOG_INFO
tqdm.pandas()
#--------------------
# class info
#--------------------
class DataSet(object):
    def __init__(self,data_dir):
        '''
            data_dir : the location of the data folder
        '''
        self.data_dir=data_dir
        
        class bangla:
            vowels                 =   ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
            consonants             =   ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 
                                        'চ', 'ছ','জ', 'ঝ', 'ঞ', 
                                        'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 
                                        'ত', 'থ', 'দ', 'ধ', 'ন', 
                                        'প', 'ফ', 'ব', 'ভ', 'ম', 
                                        'য', 'র', 'ল', 'শ', 'ষ', 
                                        'স', 'হ','ড়', 'ঢ়', 'য়']
            modifiers              =   ['ঁ', 'ং', 'ঃ','ৎ']
            # diacritics
            vowel_diacritics       =   ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
            consonant_diacritics   =   ['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
            # special charecters
            nukta                  =   '়'
            hosonto                =   '্'
            special_charecters     =   [ nukta, hosonto,'\u200d']
            
            number_values          =    ['০','১','২','৩','৪','৫','৬','৭','৮','৯']

            mods=['ঁ', 'ং', 'ঃ']

            
            sel_puntcs             =   [',', '-', '.', '/', ':','(', ')']

            class graphemes:
                dir   =   os.path.join(data_dir,"bangla","graphemes")
                csv   =   os.path.join(data_dir,"bangla","graphemes.csv")
            class numbers:
                dir   =   os.path.join(data_dir,"bangla","numbers")
                csv   =   os.path.join(data_dir,"bangla","numbers.csv")
            dict_csv  =   os.path.join(data_dir,"bangla","dictionary.csv")
            fonts     =   os.path.join(data_dir,"bangla","fonts")
        
        class english:
            class graphemes:
                dir   =   os.path.join(data_dir,"english","graphemes")
                csv   =   os.path.join(data_dir,"english","graphemes.csv")
            class numbers:
                dir   =   os.path.join(data_dir,"english","numbers")
                csv   =   os.path.join(data_dir,"english","numbers.csv")
            dict_csv  =   os.path.join(data_dir,"english","dictionary.csv")
            fonts     =   os.path.join(data_dir,"english","fonts")
        
        class common:
            class symbols:
                dir   =   os.path.join(data_dir,"common","symbols")
                csv   =   os.path.join(data_dir,"common","symbols.csv")
            class noise:
                random=   os.path.join(data_dir,"common","noise","random")
                sign  =   os.path.join(data_dir,"common","noise","signature")
            background=   os.path.join(data_dir,"common","background")  


        # assign
        self.bangla  = bangla
        self.english = english
        self.common  = common
        self.style_paths=[_path for _path in tqdm(glob(os.path.join(data_dir,"styles")))]
        # error check
        self.__checkExistance()

        # get dfs
        self.bangla.graphemes.df=self.__getDataFrame(self.bangla.graphemes)
        self.bangla.numbers.df  =self.__getDataFrame(self.bangla.numbers)
        #self.bangla.dictionary  =self.__getDataFrame(self.bangla.dict_csv,is_dict=True)

        self.english.graphemes.df=self.__getDataFrame(self.english.graphemes)
        self.english.numbers.df  =self.__getDataFrame(self.english.numbers,int_label=True)
        #self.english.dictionary  =self.__getDataFrame(self.english.dict_csv,is_dict=True)
        
        self.common.symbols.df   =self.__getDataFrame(self.common.symbols)
        # data validity
        self.__checkDataValidity(self.bangla.graphemes,"bangla.graphemes")
        self.__checkDataValidity(self.bangla.numbers,"bangla.numbers")
        self.__checkDataValidity(self.english.graphemes,"english.graphemes")
        self.__checkDataValidity(self.english.numbers,"english.numbers")
        self.__checkDataValidity(self.common.symbols,"common.symbols")
        
        self.__checkDataValidity(self.bangla.fonts,"bangla.fonts",check_dir_only=True)
        self.__checkDataValidity(self.english.fonts,"english.fonts",check_dir_only=True)
        self.__checkDataValidity(self.common.background,"common.background",check_dir_only=True)
        self.__checkDataValidity(self.common.noise.random,"common.noise.random",check_dir_only=True)
        self.__checkDataValidity(self.common.noise.sign,"common.noise.sign",check_dir_only=True)
        
        
        # graphemes
        self.bangla_graphemes=sorted(list(self.bangla.graphemes.df.label.unique()))
        
        
        
        
        
        
    def __getDataFrame(self,obj,is_dict=False,int_label=False):
        '''
            creates the dataframe from a given csv file
            args:
                obj       =   csv file path or class 
                is_dict   =   only true if the given is a dictionary 
                int_label =   if the labels are int convert string
        '''
        try:
            
            if is_dict:
                df=pd.read_csv(obj)
                assert "word" in df.columns,f"word column not found:{obj}"
                assert "graphemes" in df.columns,f"graphemes column not found:{obj}"
                 
                LOG_INFO(f"Processing Dictionary:{obj}")
                df.graphemes=df.graphemes.progress_apply(lambda x: literal_eval(x))
            else:
                csv=obj.csv
                img_dir=obj.dir
                df=pd.read_csv(csv)
                assert "filename" in df.columns,f"filename column not found:{csv}"
                assert "label" in df.columns,f"label column not found:{csv}"
                df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(img_dir,f"{x}.bmp"))
                if int_label:
                    LOG_INFO("converting int labels to string")
                    df.label=df.label.progress_apply(lambda x: str(x))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 
                

    def __checkDataValidity(self,obj,iden,check_dir_only=False):
        '''
            checks that a folder does contain proper images
        '''
        try:
            LOG_INFO(iden)
            if check_dir_only:
                data=[data_path for data_path in tqdm(glob(os.path.join(obj,"*.*")))]
                assert len(data)>0, f"No data paths found({iden})"
            else:
                imgs=[img_path for img_path in tqdm(glob(os.path.join(obj.dir,"*.*")))]
                assert len(imgs)>0, f"No data paths found({iden})"
                assert len(imgs)==len(obj.df), f"Image paths doesnot match labels data({iden}:{len(imgs)}!={len(obj.df)})"
                
        except Exception as e:
            LOG_INFO(f"Error in Validity Check:{iden}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red")        


    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.bangla.graphemes.dir),"Bangla graphemes dir not found"
        assert os.path.exists(self.bangla.graphemes.csv),"Bangla graphemes csv not found"
        assert os.path.exists(self.bangla.numbers.dir),"Bangla numbers dir not found"
        assert os.path.exists(self.bangla.numbers.csv),"Bangla numbers csv not found"
        assert os.path.exists(self.bangla.fonts),"Bangla fonts not found"
        assert os.path.exists(self.bangla.dict_csv),"Bangla dictionary not found"
        
        
        assert os.path.exists(self.english.graphemes.dir),"english graphemes dir not found"
        assert os.path.exists(self.english.graphemes.csv),"english graphemes csv not found"
        assert os.path.exists(self.english.numbers.dir),"english numbers dir not found"
        assert os.path.exists(self.english.numbers.csv),"english numbers csv not found"
        assert os.path.exists(self.english.fonts),"english fonts not found"
        assert os.path.exists(self.english.dict_csv),"english dictionary not found"
                

        assert os.path.exists(self.common.symbols.dir),"Common Symbols dir not found"
        assert os.path.exists(self.common.symbols.csv),"Common Symbols csv not found"
        assert os.path.exists(self.common.noise.random),"Random Noise dir not found"
        assert os.path.exists(self.common.noise.sign),"Sign Noise dir not found"
        assert os.path.exists(self.common.background),"Background dir not found"

        LOG_INFO("All paths found",mcolor="green")
        
        
        
        
        


    
        