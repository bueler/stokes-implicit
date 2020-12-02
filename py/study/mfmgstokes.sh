#!/bin/bash
set -e
set +x

# run:
#   $ ./mfmgstokes.sh &> results/mfmgstokes.txt

# run optimality study with 8 processes on meshes 24x24, ..., 3072x3072 (<-- N=8.5e7)
# result: all levels have 38 to 41 KSP iterations, CLEAR optimality, CLEAR optimal memory, good memory usage?

NP=8

# inputs: $1 = level number
function runcase() {
    CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../mfmgstokes.py -m0 12 -refine $1 -memory_view -log_view"
    echo
    echo $CMD
    TNAME=mfmg_lev$1.txt
    rm -f $TNAME
    $CMD &> $TNAME
    head -n 7 $TNAME            # first 7 lines are before -log_view
    grep "Time (sec):" $TNAME
}

for LEV in 1 2 3 4 5 6 7 8; do
    runcase $LEV
done

