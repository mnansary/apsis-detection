## scripts/boise_state.py
* change directory: ```cd scripts``` while executing this script 

```python 
usage: Boise State to Craft Dataset Creation Script [-h] [--height HEIGHT] [--width WIDTH] readme_txt_path save_path

positional arguments:
  readme_txt_path  Path to The **README.txt** file under **bs**folder
  save_path        Path to save the processed data

optional arguments:
  -h, --help       show this help message and exit
  --height HEIGHT  height dimension of the image : default=1024
  --width WIDTH    width dimension of the image : default=1024
```


## scripts/synthetic.py

* change directory: ```cd scripts``` while executing this script 

```python

usage: Scenetext Detection Dataset Creation Script [-h] [--train_samples TRAIN_SAMPLES] [--test_samples TEST_SAMPLES] [--cfg_data_dim CFG_DATA_DIM] [--cfg_comp_dim CFG_COMP_DIM]
                                                   [--cfg_min_num_lines CFG_MIN_NUM_LINES] [--cfg_max_num_lines CFG_MAX_NUM_LINES] [--cfg_min_num_words CFG_MIN_NUM_WORDS]
                                                   [--cfg_max_num_words CFG_MAX_NUM_WORDS] [--cfg_min_word_len CFG_MIN_WORD_LEN] [--cfg_max_word_len CFG_MAX_WORD_LEN] [--cfg_min_num_len CFG_MIN_NUM_LEN]
                                                   [--cfg_max_num_len CFG_MAX_NUM_LEN] [--cfg_word_min_space CFG_WORD_MIN_SPACE] [--cfg_word_max_space CFG_WORD_MAX_SPACE]
                                                   [--cfg_vert_min_space CFG_VERT_MIN_SPACE] [--cfg_vert_max_space CFG_VERT_MAX_SPACE] [--cfg_heatmap_ratio CFG_HEATMAP_RATIO]
                                                   [--cfg_languages CFG_LANGUAGES [CFG_LANGUAGES ...]] [--cfg_formats CFG_FORMATS [CFG_FORMATS ...]] [--cfg_components CFG_COMPONENTS [CFG_COMPONENTS ...]]
                                                   data_dir save_dir format dataset_iden

positional arguments:
  data_dir              Path of the source data folder
  save_dir              Path of the directory to save the dataset
  format                The desired format for creating the data. Available:totaltext,linetext
  dataset_iden          The desired name for the dataset.Use something that can help you remember the generation details.Example: (bangla_synth) may indicate only bangla data

optional arguments:
  -h, --help            show this help message and exit
  --train_samples TRAIN_SAMPLES
                        number of train samples to create : default=1500
  --test_samples TEST_SAMPLES
                        number of test samples to create : default=128
  --cfg_data_dim CFG_DATA_DIM
                        dimension of the image [Since only squre images are produced, providing one value is enough] : default=1024
  --cfg_comp_dim CFG_COMP_DIM
                        height dimension for any kind of component : default=64
  --cfg_min_num_lines CFG_MIN_NUM_LINES
                        min number of lines per image : default=1
  --cfg_max_num_lines CFG_MAX_NUM_LINES
                        max number of lines per image : default=10
  --cfg_min_num_words CFG_MIN_NUM_WORDS
                        min number of words per line : default=1
  --cfg_max_num_words CFG_MAX_NUM_WORDS
                        max number of words per line : default=10
  --cfg_min_word_len CFG_MIN_WORD_LEN
                        min number of components in a word if component type:[grapheme] is used : default=1
  --cfg_max_word_len CFG_MAX_WORD_LEN
                        max number of components in a word if component type:[grapheme] is used : default=10
  --cfg_min_num_len CFG_MIN_NUM_LEN
                        min number of digits in a number if component type:[number] is used : default=1
  --cfg_max_num_len CFG_MAX_NUM_LEN
                        max number of digits in a number if component type:[number] is used : default=10
  --cfg_word_min_space CFG_WORD_MIN_SPACE
                        min space between two words/numbers [in pixels] : default=50
  --cfg_word_max_space CFG_WORD_MAX_SPACE
                        max space between two words/numbers [in pixels] : default=100
  --cfg_vert_min_space CFG_VERT_MIN_SPACE
                        min space between two lines [in pixels] : default=1
  --cfg_vert_max_space CFG_VERT_MAX_SPACE
                        max space between two lines [in pixels] : default=100
  --cfg_heatmap_ratio CFG_HEATMAP_RATIO
                        heatmap distance ration if line text format is used[float available] : default=2
  --cfg_languages CFG_LANGUAGES [CFG_LANGUAGES ...]
                        a list of language source to be used|| available:[bangla,english]. e.g., "--cfg_languages bangla englist", or "--cfg_languages bangla" (for single use)
  --cfg_formats CFG_FORMATS [CFG_FORMATS ...]
                        a list of formats to be used ||available:[handwriten,printed]. e.g., "--cfg_formats handwriten printed"
  --cfg_components CFG_COMPONENTS [CFG_COMPONENTS ...]
                        a list of type of components to be used ||available:["number","grapheme","mixed"]. e.g., "--cfg_components number grapheme"


```

## scripts/memo.py 
* change directory: ```cd scripts``` while executing this script 

```python
usage: Synthetic Memo Data Creation Script [-h] [--height HEIGHT] [--n_data N_DATA] data_dir save_dir

positional arguments:
  data_dir         Path to base data under source
  save_dir         Path to save the processed data

optional arguments:
  -h, --help       show this help message and exit
  --height HEIGHT  height dimension of the image : default=1024
  --n_data N_DATA  number of data to create : default=1024
```

## scripts/store.py 
* change directory: ```cd scripts``` while executing this script 

```python
usage: TFRecords Data Creation Script [-h] data_dir save_dir ds_iden

positional arguments:
  data_dir    Path to the images folder
  save_dir    Path to save the tfrecords
  ds_iden     dataset Identifier

optional arguments:
  -h, --help  show this help message and exit
```

## scripts/craftsynth.py 


```python

usage: Scenetext Detection Dataset Creation Script [-h] [--train_samples TRAIN_SAMPLES] [--cfg_data_dim CFG_DATA_DIM] [--cfg_comp_dim CFG_COMP_DIM] [--cfg_min_num_lines CFG_MIN_NUM_LINES]
                                                   [--cfg_max_num_lines CFG_MAX_NUM_LINES] [--cfg_min_num_words CFG_MIN_NUM_WORDS] [--cfg_max_num_words CFG_MAX_NUM_WORDS] [--cfg_min_word_len CFG_MIN_WORD_LEN]
                                                   [--cfg_max_word_len CFG_MAX_WORD_LEN] [--cfg_min_num_len CFG_MIN_NUM_LEN] [--cfg_max_num_len CFG_MAX_NUM_LEN] [--cfg_word_min_space CFG_WORD_MIN_SPACE]
                                                   [--cfg_word_max_space CFG_WORD_MAX_SPACE] [--cfg_vert_min_space CFG_VERT_MIN_SPACE] [--cfg_vert_max_space CFG_VERT_MAX_SPACE]
                                                   [--cfg_heatmap_ratio CFG_HEATMAP_RATIO] [--cfg_languages CFG_LANGUAGES [CFG_LANGUAGES ...]] [--cfg_formats CFG_FORMATS [CFG_FORMATS ...]]
                                                   [--cfg_components CFG_COMPONENTS [CFG_COMPONENTS ...]]
                                                   data_dir save_dir dataset_iden

positional arguments:
  data_dir              Path of the base folder under source data folder
  save_dir              Path of the directory to save the dataset
  dataset_iden          The desired name for the dataset.Use something that can help you remember the generation details.Example: (bangla_synth) may indicate only bangla data

optional arguments:
  -h, --help            show this help message and exit
  --train_samples TRAIN_SAMPLES
                        number of train samples to create : default=10000
  --cfg_data_dim CFG_DATA_DIM
                        dimension of the image [Since only squre images are produced, providing one value is enough] : default=786
  --cfg_comp_dim CFG_COMP_DIM
                        height dimension for any kind of component : default=64
  --cfg_min_num_lines CFG_MIN_NUM_LINES
                        min number of lines per image : default=1
  --cfg_max_num_lines CFG_MAX_NUM_LINES
                        max number of lines per image : default=10
  --cfg_min_num_words CFG_MIN_NUM_WORDS
                        min number of words per line : default=1
  --cfg_max_num_words CFG_MAX_NUM_WORDS
                        max number of words per line : default=10
  --cfg_min_word_len CFG_MIN_WORD_LEN
                        min number of components in a word if component type:[grapheme] is used : default=1
  --cfg_max_word_len CFG_MAX_WORD_LEN
                        max number of components in a word if component type:[grapheme] is used : default=10
  --cfg_min_num_len CFG_MIN_NUM_LEN
                        min number of digits in a number if component type:[number] is used : default=1
  --cfg_max_num_len CFG_MAX_NUM_LEN
                        max number of digits in a number if component type:[number] is used : default=10
  --cfg_word_min_space CFG_WORD_MIN_SPACE
                        min space between two words/numbers [in pixels] : default=50
  --cfg_word_max_space CFG_WORD_MAX_SPACE
                        max space between two words/numbers [in pixels] : default=100
  --cfg_vert_min_space CFG_VERT_MIN_SPACE
                        min space between two lines [in pixels] : default=1
  --cfg_vert_max_space CFG_VERT_MAX_SPACE
                        max space between two lines [in pixels] : default=100
  --cfg_heatmap_ratio CFG_HEATMAP_RATIO
                        heatmap distance ration if line text format is used[float available] : default=2
  --cfg_languages CFG_LANGUAGES [CFG_LANGUAGES ...]
                        a list of language source to be used|| available:[bangla,english]. e.g., "--cfg_languages bangla englist", or "--cfg_languages bangla" (for single use)
  --cfg_formats CFG_FORMATS [CFG_FORMATS ...]
                        a list of formats to be used ||available:[handwriten,printed]. e.g., "--cfg_formats handwriten printed"
  --cfg_components CFG_COMPONENTS [CFG_COMPONENTS ...]
                        a list of type of components to be used ||available:["number","grapheme","mixed"]. e.g., "--cfg_components number grapheme"

```