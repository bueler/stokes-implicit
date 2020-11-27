#!/bin/bash
set -e
set +x

# run as:  ./pool_stage3.sh &> results/pool3.txt

# run optimality study with 8 processes for stage 3
# meshes: 4x4x4, 8x8x8, 16x16x16, 32x32x32, 64x64x64

NP=8

# inputs: $1 = level number
#         $2 = mesh specification
function runcase() {
    CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../pool.py -stage 3 $2 -log_view"
    echo
    echo $CMD
    TNAME=stage3_lev$1.txt
    rm -f $TNAME
    $CMD &> $TNAME
    head -n 6 $TNAME            # first 6 lines are before -log_view
    grep "PCSetUp " $TNAME
    grep "KSPSolve " $TNAME
    grep "Time (sec):" $TNAME
}

runcase 0 "-mx 4 -my 4 -refine 1 -aggressive"
runcase 1 "-mx 8 -my 8 -mz 2 -refine 1 -aggressive"
runcase 2 "-mx 16 -my 16 -refine 2 -aggressive"
runcase 3 "-mx 32 -my 32 -mz 2 -refine 2 -aggressive"
runcase 4 "-mx 64 -my 64 -refine 3 -aggressive"

