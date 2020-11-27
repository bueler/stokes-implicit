#!/bin/bash
set -e
set +x

# run as:  ./pool_stage1.sh &> results/pool1.txt

# run optimality study with 8 processes for stage 1
# meshes: 4x4x4, 8x8x8, 16x16x16, 32x32x32, 64x64x64

NP=8

for LEV in 0 1 2 3 4; do
    echo
    CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../pool.py -stage 1 -mx 4 -my 4 -mz 4 -refine $LEV -log_view"
    echo $CMD
    TNAME=stage1_lev$LEV.txt
    rm -f $TNAME
    $CMD &> $TNAME
    head -n 6 $TNAME
    grep "PCSetUp " $TNAME
    grep "KSPSolve " $TNAME
    grep "Time (sec):" $TNAME
done
