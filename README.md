# banglaCraft
banga Dataset for CRAFTS and CRAFT
```python
Version: 0.0.1     
Authors: Md. Nazmuddoha Ansary, 
```
**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
```
# Setup
>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
*  ```sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools```
* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**

* ```pip3 install pillow --global-option="build_ext" --global-option="--enable-freetype"```
* ```pip3 install -r requirements.txt``` 
> Its better to use a virtual environment 
# Dataset
* **caution**: Do not change anything under **resources**. The directory should look as follows
```python
    ├── graphemes.csv
    └── numbers.csv
```
* The **grapheme** dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
    * Only the **256** folder under **256_train** is kept and renamed as **RAW** form **BengaliAI:Supplementary dataset for BengaliAI Competition**
* The **number** dataset is taken from [here](https://www.kaggle.com/nazmuddhohaansary/banglasymbols) 
    * Only the **RAW_NUMS** folder is kept that contains all the images of the numbers
* The final **data** folder is like as follows:

```python
    data
    ├── RAW_NUMS
    └── RAW
```


# ISSUES FACED
* **UpsampleLike** layer is tf-1 compat : https://github.com/tensorflow/tensorflow/issues/45207