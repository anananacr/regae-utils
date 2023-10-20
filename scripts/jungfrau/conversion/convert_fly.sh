#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=40G

#SBATCH --job-name  convert_jf
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_6-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_6-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh my-file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

INPUT=221115_mica_6/221115_mica_6
LABEL=20221115_mica_6
DATE=20221115
PEDAL=20221115_mica_6
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted

python3 convert_images.py -p1 ${ROOT}/pedal/pedal_d0_${PEDAL}.h5 -p2 ${ROOT}/pedal/pedal_d1_${PEDAL}.h5 -g1 ${ROOT}/pedal/gainMaps_M283.bin -g2 ${ROOT}/pedal/gainMaps_M281.bin -i ${ROOT}/../../../raw/${INPUT} -s 1 -e 2 -o ${ROOT}/mica_6/mica_6