#!/bin/bash

#SBATCH --job-name=myjob

#SBATCH --account=stf
#SBATCH --partition=stf

#SBATCH --nodes=1

#SBATCH --time=1:00:00

#SBATCH --mem=50G

#SBATCH --workdir=/gscratch/stf/ducela/ETSscripts/longrange

module load anaconda3_5.3
python compute_Plourde.py

