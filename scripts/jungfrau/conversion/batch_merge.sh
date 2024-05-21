#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=10G

#SBATCH --job-name  merge_chunks
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/converted/error/mica-%N-%j.err

# Author: Ana Carolina Rodrigues (2021-2024)
# Email: ana.rodrigues@desy.de
# Description: Concatenate h5 files from same run measured in step mode, size of chunks hard coded in merge_chunks.py
# Usage: sbatch batch_merge.sh 
 
source /home/rodria/scripts/regae/env-regae/bin/activate
python merge_chunks.py -i /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/240516_C3b_MICA020_3GHz_002/ed_rot_step_001/240516_C3b_MICA020_3GHz_002_001_master -s 0 -e 0 -o /asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted/240516_C3b_MICA020_3GHz_002/ed_rot_step_001/240516_C3b_MICA020_3GHz_002_001