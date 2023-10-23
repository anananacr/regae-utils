#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=4G

#SBATCH --job-name  convert_jf
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh my-file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/231020_mica_c4_m1_001/ed_rot_step_001 
#ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/231020_mica_c4_m1_001/ed_rot_step_001
SCRATCH=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted
for i in $(seq 0 1 1001); do
    NEXT=$((i+1))
    python3 apply_geom.py -i ${ROOT}/231020_mica_c4_m1_001_001 -g ${SCRATCH}/pedal/JF_regae.geom -s ${i} -e ${NEXT} -o ${SCRATCH}/231020_mica_c4_m1_001_pad_average/231020_mica_c4_m1_001_pad_average_${i};
done;
