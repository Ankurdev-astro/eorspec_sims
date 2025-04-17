#!/bin/bash
export OMP_NUM_THREADS=2

echo "$(which python)"
echo "$(python --version)"

mpirun -np 64 python write_toast_maps.py