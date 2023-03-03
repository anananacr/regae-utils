#!/bin/sh

INPUT=$1
DATE=20220524
LABEL=$2
PEDAL_DATE=20220524
STEP=100

for i in $(seq 34 1 34); do
    mkdir ../proc/centered/${DATE}/; for j in $(seq 100 ${STEP} 150); do
        python3 sumup_nocenter.py -p1 ../calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ../calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ../calib/gainMaps_M283.bin -g2 ../calib/gainMaps_M281.bin -i ${INPUT}_master_${i}.h5 -m ../calib/mask_v0.h5 -n ${STEP} -b $j -o ../proc/centered/${DATE}/${DATE}_${LABEL}_${i}_${j}; done; done;
