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

# ISSUES FACED
* **UpsampleLike** layer is tf-1 compat : https://github.com/tensorflow/tensorflow/issues/45207