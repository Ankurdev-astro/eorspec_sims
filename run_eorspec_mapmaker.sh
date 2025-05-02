#!/bin/bash
export OMP_NUM_THREADS=2

echo "$(which python)"
echo "$(python --version)"

###################
####  CONFIG   ####
###################
CHANNEL=340
STEP='step216'
INDIR='data_CII_tomo'
OUTDIR='outmaps_fb_v5'
###################

mpirun -np 48 python write_toast_maps.py -c $CHANNEL --step $STEP -in $INDIR -out $OUTDIR