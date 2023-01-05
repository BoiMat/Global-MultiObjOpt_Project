#!/bin/bash
#PBS -q dssc
#PBS -l nodes=1:ppn=24
#PBS -l walltime=03:00:00
#PBS -N serial

cd $PBS_O_WORKDIR/

module load conda/4.9.2
conda activate Global

start=`date +%s`

python test.py

end=`date +%s`

runtime=$((end-start))
