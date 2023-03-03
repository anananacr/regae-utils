#!/bin/sh

INPUT=$1
LABEL=$2

DATE=20220524

for i in $(seq 0 1 82); do
    mkdir ../proc/average/${DATE}/;
    python3 merge_average.py -i ${INPUT}_${i} -o ../proc/average/${DATE}/${DATE}_${LABEL}_${i}; done;
