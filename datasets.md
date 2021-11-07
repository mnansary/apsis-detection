# Base Dataset
* The overall dataset is available here: https://www.kaggle.com/nazmuddhohaansary/sourcedata
* The folder structre should look as follows:

```python
        base
        ├── bangla
        │   ├── graphemes.csv
        │   ├── numbers.csv
        │   ├── dictionary.csv
        │   ├── fonts
        │   ├── graphemes
        │   └── numbers
        ├── common
        │   ├── symbols.csv
        │   ├── background
        │   ├── noise
        │   │   ├── random
        │   │   └── signature
        │   └── symbols
        └── english
            ├── graphemes.csv
            ├── numbers.csv
            ├── dictionary.csv
            ├── fonts
            ├── graphemes
            └── numbers    
```

# Natrual Dataset
## Boise State
* **Boise_State_Bangla_Handwriting_Dataset_20200228.zip**  from  [**Boise State Bangla Handwriting Dataset**](https://scholarworks.boisestate.edu/saipl/1/)
* **Instructions**: 
  * unzip the file
  * corrupted zip issue:**fix zip issues with zip -FFv if needed**
  * rename the unzipped folder as **bs**
  * The **bs** folder structre should be as follows:
  
```
    bs
    ├── 1. Camera
    ├── 2. Scan
    ├── 3. Conjunct
    ├── 4. Demographic.txt
    └── README.txt
```
## ICDAR-2013
* **Task 2.2: Text Segmentation (2013 edition)** files from [ICDAR-Challenges](https://rrc.cvc.uab.es/?ch=2&com=downloads)
* **Instrauctions**
    * unzip **Challenge2_Training_Task12_Images.zip** and rename the extracted folder as **Images**
    * unzip **Challenge2_Training_Task2_GT.zip** and rename the extracted folder as **GT**
    * put **Images** and **GT** under a folder named **icdar**
    * The **icdar** folder structre should look as follows:

```
    icdar
    ├── GT
    │   ├── 100_GT.bmp
    │   ├── 100_GT.txt
    │   ├── 101_GT.bmp
    │   ├── 101_GT.txt
    │   ...................
    │   ├── 328_GT.bmp
    │   └── 328_GT.txt
    └── Images
        ├── 100.jpg
        ├── 101.jpg
        ├── 102.jpg
        ......................
        ......................
```

# Final Data Source Directory Structre:

```
source
├── base
│   ├── bangla
│   ├── common
│   └── english
├── natrural
│   ├── bs
│   └── icdar
└── styles
    └── memo
```