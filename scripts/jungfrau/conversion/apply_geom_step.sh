#!/bin/sh

#SBATCH --partition=cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=4G

#SBATCH --job-name  convert_jf
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_5-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/error/mica_5-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh my-file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

INPUT=221115_mica_5/221115_mica_5
LABEL=20221115_mica_5
DATE=20221115
PEDAL=20221115_mica_5
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted

for i in $(seq 0 1 1481); do
    NEXT=$((i+1))
    python3 apply_geom.py -i ${ROOT}/mica_5/mica_5 -g ${ROOT}/pedal/JF_regae_221115_Au.geom -s ${i} -e ${NEXT} -o ${ROOT}/mica_5_pad/mica_5_${i};
done;
