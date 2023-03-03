id=1
python3 sumup_nodark.py -p1 ../calib/pedal_d0_20220421.h5 -p2 ../calib/pedal_d1_20220421.h5 -g1 ../calib/gainMaps_M283.bin -g2 ../calib/gainMaps_M281.bin -i /gpfs/current/raw/20220404_ni_on_master_$id.h5 -m ../calib/mask_v0.h5 -f 0 -n 100 -b 0 -o ../proc/centered/20220421_ni_on_$id
