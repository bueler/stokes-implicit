#!/bin/bash
set -e
set +x

# run for stages 2 and 3:
#   $ ./pool_stage23.sh 2 &> results/pool2.txt
#   $ ./pool_stage23.sh 3 &> results/pool3.txt

# run optimality study with 8 processes for stage 2
# meshes all have 1 base layer and each has 4 times as many elements as the last:
#     4x4x4, 8x8x4, 16x16x4, 16x16x16, 32x32x16, 64x64x16, 64x64x64

STAGE=$1
NP=8

# inputs: $1 = level number
#         $2 = mesh specification
function runcase() {
    CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../pool.py -stage $STAGE $2 -log_view"
    echo
    echo $CMD
    TNAME=stage2_lev$1.txt
    rm -f $TNAME
    $CMD &> $TNAME
    head -n 7 $TNAME            # first 7 lines are before -log_view
    grep "PCSetUp " $TNAME
    grep "KSPSolve " $TNAME
    grep "Time (sec):" $TNAME
}

runcase 0 "-mx 4 -my 4 -refine 1 -aggressive"
runcase 1 "-mx 8 -my 8 -refine 1 -aggressive"
runcase 2 "-mx 16 -my 16 -refine 1 -aggressive"
runcase 3 "-mx 16 -my 16 -refine 2 -aggressive"
runcase 4 "-mx 32 -my 32 -refine 2 -aggressive"
runcase 5 "-mx 64 -my 64 -refine 2 -aggressive"
runcase 6 "-mx 64 -my 64 -refine 3 -aggressive"

