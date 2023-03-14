#!/bin/sh

INPUT=$1
LABEL=$2

DATE=20220524

for i in $(seq 0 1 82); do
    mkdir ../proc/merged/${DATE}/;
    python3 merge_scan.py -i ${INPUT}_${i}_ -o ../proc/merged/${DATE}/${DATE}_${LABEL}_${i}; done;
