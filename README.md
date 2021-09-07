# synthdata

```python
Version: 0.0.5     
Authors: Md. Nazmuddoha Ansary,
         Md. Rezwanul Haque,
         Md. Mobassir Hossain 
```
**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
```
# Environment Setup

>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
>
>*  ```sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools```

* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**

* **pip requirements**: ```pip install -r requirements.txt``` 

> Its better to use a virtual environment 
> OR use conda-

* **conda**: use environment.yml: ```conda env create -f environment.yml```



# Dataset Processing:
* checkout **datasets.md** for downloading resources and **management instructions**
* checkout **scripts.md** for executing processing on various datasets
* each script is executable separately 
* for combined execution use **scripts/server.sh** if your data folder are stored according to **datasets.md**
```bash
#!/bin/sh
DATASET_MAIN_PATH="/media/ansary/DriveData/Work/APSIS/datasets/Detection/"
BS_README_TXT_PATH="${DATASET_MAIN_PATH}source/natrural/bs/README.txt"
PROCESSED_PATH="${DATASET_MAIN_PATH}processed/"
BASE_DATA_PATH="${DATASET_MAIN_PATH}source/base/"
ICDAR_DATA_PATH="${DATASET_MAIN_PATH}source/natrural/icdar/Images/"
# icdar
python icdar.py $ICDAR_DATA_PATH $PROCESSED_PATH
python store.py "${PROCESSED_PATH}icdar/images/" $DATASET_MAIN_PATH icdar 
# memo
python memo.py $BASE_DATA_PATH $PROCESSED_PATH
python store.py "${PROCESSED_PATH}memo/images/" $DATASET_MAIN_PATH memo 
# boise state
python boise_state.py $BS_README_TXT_PATH $PROCESSED_PATH
python store.py "${PROCESSED_PATH}bs/images/" $DATASET_MAIN_PATH bs 
# synthetic
python synthetic.py $BASE_DATA_PATH $PROCESSED_PATH linetext synth --cfg_heatmap_ratio 4
python store.py "${PROCESSED_PATH}synth.train/images/" $DATASET_MAIN_PATH synth.train 
python store.py "${PROCESSED_PATH}synth.test/images/" $DATASET_MAIN_PATH synth.test 
echo succeded
```


**Datasets Used**
- [x] Boise-State bangla
- [x] Synthetic Mixed language data
- [x] Memo(Table) Data
- [x] ICDAR-2013 Dataset
