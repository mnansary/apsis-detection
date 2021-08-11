#!/bin/sh
python /home/apsisdev/ansary/CODES/synthdata/memo_data.py && \
python /home/apsisdev/ansary/CODES/synthdata/store.py  && \
zip -r /home/apsisdev/ansary/DATASETS/Detection/memo_table/noisy/noisy.zip  /home/apsisdev/ansary/DATASETS/Detection/memo_table/noisy/tfrecords/
echo succeeded
