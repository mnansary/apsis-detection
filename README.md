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

**Datasets Used**
- [x] Boise-State bangla
- [x] Synthetic Mixed language data
- [x] Memo(Table) Data
- [ ] ICDAR-2013 Dataset
