## Example of running sumup_nodark.py on fast mode defined by calculating the center of mass. Set correct path to gain maps pedestals converted and mask file

## Set file index
id =1
## Set n frames to be summed
n = 100
## Set beginnning frame
b = 0

python3 sumup_nodark.py -p1 ../calib/pedal_d0_20220421.h5 -p2 ../calib/pedal_d1_20220421.h5 -g1 ../calib/gainMaps_M283.bin -g2 ../calib/gainMaps_M281.bin -i /path/to/file/file_$id.h5 -m ../calib/mask_v0.h5 -f 0 -n $n -b $b -o /path/to/output/file/centered/20220421_ni_on_$id
