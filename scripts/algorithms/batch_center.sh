#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --partition=allcpu,cfel
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=100
#SBATCH --mem=60G
#SBATCH --job-name  center_fly
#SBATCH --output    /gpfs/cfel/user/rodria/processed/regae/2022/error/cc-%N-%j.out
#SBATCH --error     /gpfs/cfel/user/rodria/processed/regae/2022/error/cc-%N-%j.err
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT
#SBATCH --mail-user=ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

INPUT=/gpfs/cfel/user/rodria/processed/regae/2022/lists/mica_4.lst
OUTPUT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/processed/mica_4

mkdir $OUTPUT;
mkdir $OUTPUT/plots;
mkdir $OUTPUT/cc_data;
mkdir $OUTPUT/plots/cc_map;
mkdir $OUTPUT/plots/cc_flip;
mkdir $OUTPUT/plots/third;

./find_center.py -i ${INPUT} -m  /gpfs/cfel/user/rodria/processed/regae/2022/lists/mask_mica.lst -o ${OUTPUT}
