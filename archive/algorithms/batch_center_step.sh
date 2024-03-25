# Split a large centering job into many small tasks and submit using SLURM

# ./batch_center_step.sh start_file_index n_files_per_job end_file_index 
# Copyright � 2016-2017 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Authors:
#      

MAIL=ana.rodrigues@desy.de

INPUT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/lists/mica_6.lst
MASK=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/lists/mask_mica.lst

START=$1
STEP=$2
END=$3
OUTPUT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/processed/mica_6
LABEL=mica_4

mkdir $OUTPUT;
mkdir $OUTPUT/plots;
mkdir $OUTPUT/cc_data;
mkdir $OUTPUT/plots/cc_map;
mkdir $OUTPUT/plots/cc_flip;
mkdir $OUTPUT/plots/third;

for i in $(seq $START $STEP $END); do
    JNAME='center-step'
    NAME='center-step'
    SLURMFILE="${NAME}_${LABEL}_${i}.sh"

    echo "#!/bin/sh" > $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --partition=allcpu,cfel" >> $SLURMFILE  # Set your partition here
    echo "#SBATCH --time=2-0:00:00" >> $SLURMFILE
    echo "#SBATCH --nodes=1" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --chdir   $PWD" >> $SLURMFILE
    echo "#SBATCH --job-name  $JNAME" >> $SLURMFILE
    echo "#SBATCH --output    /gpfs/cfel/user/rodria/processed/regae/2022/error/${NAME}-${LABEL}-%N-%j.out" >> $SLURMFILE
    echo "#SBATCH --error     /gpfs/cfel/user/rodria/processed/regae/2022/error/${NAME}-${LABEL}-%N-%j.err" >> $SLURMFILE
    echo "#SBATCH --nice=128" >> $SLURMFILE
    echo "#SBATCH --mincpus=100" >> $SLURMFILE
    echo "#SBATCH --mem=10G" >> $SLURMFILE
    echo "#SBATCH --mail-type FAIL,REQUEUE,TIME_LIMIT" >> $SLURMFILE
    echo "#SBATCH --mail-user $MAIL" >> $SLURMFILE
    echo >> $SLURMFILE

    echo "source /etc/profile.d/modules.sh" >> $SLURMFILE
    echo "module load maxwell python/3.7" >> $SLURMFILE
    echo "source /home/rodria/scripts/regae/env-regae/bin/activate" >> $SLURMFILE

    END_FILE=$((i+STEP))

    command="./find_center.py -i ${INPUT} -m ${MASK} -s ${i} -e ${END_FILE} -o ${OUTPUT}"

    echo $command >> $SLURMFILE

    echo "chmod a+rw $PWD" >> $SLURMFILE

    sbatch $SLURMFILE
    mv $SLURMFILE /asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/converted/shell;
done
