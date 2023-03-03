#!/bin/sh

INPUT=$1
DATE=20220524
LABEL=$2
PEDAL_DATE=20220524
STEP=100
NORM=../proc/centered/20220524/20220524_Au30_0_0.h5

mkdir ../proc/centered/${DATE}/;
mkdir ../proc/scaled/${DATE}/;
mkdir ../proc/merged/${DATE}/;
mkdir ../proc/average/${DATE}/;

for i in $(seq 0 1 82); do
    for j in $(seq 0 ${STEP} 400); do
        python3 sumup_nocenter.py -p1 ../calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ../calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ../calib/gainMaps_M283.bin -g2 ../calib/gainMaps_M281.bin -i ${INPUT}_master_${i}.h5 -m ../calib/mask_v0.h5 -n ${STEP} -b $j -o ../proc/centered/${DATE}/${DATE}_${LABEL}_${i}_${j};
        python3 scaling.py -i ../proc/centered/${DATE}/${DATE}_${LABEL}_${i}_${j}.h5 -n  ${NORM} -o ../proc/scaled/${DATE}/${DATE}_${LABEL}_${i}_${j}; done;
    python3 merge_scan.py -i ../proc/scaled/${DATE}/${DATE}_${LABEL}_${i}_ -o ../proc/merged/${DATE}/${DATE}_${LABEL}_${i}; 
    python3 merge_average.py -i ../proc/merged/${DATE}/${DATE}_${LABEL}_${i} -o ../proc/average/${DATE}/${DATE}_${LABEL}_${i}; done;
python3 scan_radial_to_file.py -i ../proc/average/${DATE}/${DATE}_${LABEL} -o ../proc/average/${DATE}/${DATE}_${LABEL}_radial


