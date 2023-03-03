#!/bin/sh

# Split a large converting job into many small tasks and submit using SLURM

# ./turbo_convert.sh my-file.h5 label

# Copyright � 2021-2023 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Authors:
# Ana Carolina Rodrigues

#MAIL=ana.rodrigues@desy.de

# Set up environment here if necessary
INPUT=$1
LABEL=$2
DATE=20221115

PEDAL_DATE=20221115_2
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614

## SET THIS PARAMETERS FOR FLY SCAN
#START=0
#END=399
#STEP=40

## SET THIS PARAMETERS FOR STEP SCAN
START=0
END=9800
STEP=100
FRAMES=10

mkdir ./shell/
mkdir ${ROOT}/processed/converted/${DATE}_${LABEL}

# Loop over the event list files, and submit a batch job for each of them
for j in $(seq $START $STEP $END); do
    JNAME='cryst_convert_regae-'
    NAME='cryst_convert_regae'
    SLURMFILE="${NAME}_${DATE}_${LABEL}_${j}.sh"
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

	command="python3 ../conversion/convert_all.py -p1 ${ROOT}/processed/calib/pedal_d0_${PEDAL_DATE}.h5 -p2 ${ROOT}/processed/calib/pedal_d1_${PEDAL_DATE}.h5 -g1 ${ROOT}/processed/calib/gainMaps_M283.bin -g2 ${ROOT}/processed/calib/gainMaps_M281.bin "
    command="$command -i ${ROOT}/raw/${INPUT}"
    command="$command -s ${j}"
    command="$command -e ${LAST}"
    command="$command -f ${FRAMES}"
    command="$command -o ${ROOT}/processed/converted/${DATE}_${LABEL}/${DATE}_${LABEL}_${j}"

    echo $command >> $SLURMFILE

    echo "chmod a+rw $PWD" >> $SLURMFILE

    sbatch $SLURMFILE
mv $SLURMFILE ./shell

done
