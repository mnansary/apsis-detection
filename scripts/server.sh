#!/bin/sh
DATASET_MAIN_PATH="/home/apsisdev/ansary/DATASETS/APSIS/Detection/"
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
python synthetic.py $BASE_DATA_PATH $PROCESSED_PATH linetext synth 
python store.py "${PROCESSED_PATH}synth.train/images/" $DATASET_MAIN_PATH synth.train 
python store.py "${PROCESSED_PATH}synth.test/images/" $DATASET_MAIN_PATH synth.test 
echo succeded