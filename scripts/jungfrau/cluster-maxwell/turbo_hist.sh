#!/bin/sh

# Take segments of a image to search for single electron counting distribution and submit using SLURM. Remember to set correctly the root path for data.

# ./turbo_hist.sh my-file.h5 

# Copyright 2021 - 2023 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Authors:
# Ana Carolina Rodrigues

#MAIL=ana.rodrigues@desy.de
INPUT=$1

ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11015669/scratch_cc/rodria/safe/

START_X=455
END_X=456
START_Y=595
END_Y=596
STEP=2

mkdir ./shell/
# Loop over the event list files, and submit a batch job for each of them
for i in $(seq $START_X $STEP $END_X); do
    for j in $(seq $START_Y $STEP $END_Y); do
        JNAME='hist_regae-'
        NAME='hist_regae'
        SLURMFILE="${NAME}_${i}_${j}.sh"
        LAST=$((j+STEP))

        echo "#!/bin/sh" > $SLURMFILE
        echo >> $SLURMFILE

        echo "#SBATCH --partition=cfel" >> $SLURMFILE  # Set your partition here
        echo "#SBATCH --time=1:00:00" >> $SLURMFILE
        echo "#SBATCH --nodes=1" >> $SLURMFILE
        echo >> $SLURMFILE

        echo "#SBATCH --chdir   $PWD" >> $SLURMFILE
        echo "#SBATCH --job-name  $JNAME" >> $SLURMFILE
        echo "#SBATCH --output    $NAME-%N-%j.out" >> $SLURMFILE
        echo "#SBATCH --error     $NAME-%N-%j.err" >> $SLURMFILE
	    echo "#SBATCH --nice=0" >> $SLURMFILE
        #echo "#SBATCH --mail-type ALL" >> $SLURMFILE
        #echo "#SBATCH --mail-user $MAIL" >> $SLURMFILE
        echo >> $SLURMFILE

	    command="python3 ../histogram/hist_pixel.py -i ${ROOT}/${INPUT} -px ${i} -py ${j} -o ${ROOT}/plots/hist_Se_10000_${j}_${i}"

        echo $command >> $SLURMFILE

        echo "chmod a+rw $PWD" >> $SLURMFILE

        sbatch $SLURMFILE
        mv $SLURMFILE ./shell;
    done;
done
