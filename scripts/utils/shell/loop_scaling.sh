#!/bin/sh

INPUT=$1
LABEL=$2
NORM=../proc/centered/20220524/20220524_Au30_30_0.h5
DATE=20220524

for i in $(seq 0 1 82); do
    mkdir ../proc/scaled/${DATE}/; for j in $(seq 0 100 400); do
        python3 scaling.py -i ${INPUT}_${i}_${j}.h5 -n  ${NORM} -o ../proc/scaled/${DATE}/${DATE}_${LABEL}_${i}_${j}; done; done;
