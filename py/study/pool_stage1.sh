#!/bin/bash
set -e
set +x

# run optimality study with 8 processes for stage 1
# meshes: 4x4x4, 8x8x8, 16x16x16, 32x32x32, 64x64x64

NP=8

for LEV in 0 1 2 3 4; do
    echo
    CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../pool.py -stage 1 -mx 4 -my 4 -mz 4 -refine $LEV -log_view"
    echo $CMD
    rm -f stage1lev$LEV.txt 
    $CMD &> stage1lev$LEV.txt
    head -n 6 stage1lev$LEV.txt
    grep "PCSetUp " stage1lev$LEV.txt
    grep "KSPSolve " stage1lev$LEV.txt
    grep "Time (sec):" stage1lev$LEV.txt
done
