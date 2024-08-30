#!/bin/bash
set -e
set +x

FIXME THINK IT THROUGH; CAN GO TO 257^3 Q1 MESHES

# compute condition number of preconditioned matrices for three stages

NP=1

function runcase() {
  CMD="mpiexec --bind-to hwthread --map-by core -n $NP ../sole.py -stage $1 -s_ksp_view_singularvalues $2 -refine $3 -log_view"
  echo $CMD
  rm -f tmp.txt
  $CMD &> tmp.txt
  head -n 6 tmp.txt
  grep "KSPSolve " tmp.txt
}

# in each stage do 6 runs:
#     4^3, 8^3, 16^3, 32^3, 64^3, 128^3

echo '***** STAGE 1 *****'
for REF in 1 2 3 4 5 6; do
    runcase 1 "-mz 2" $REF
done

for ST in 2 3 4; do
    echo
    echo '***** STAGE' $ST '*****'
    MX=4
    MY=4
    for REF in 1 2; do
        runcase $ST "-mx $MX -my $MY" $REF
        ((MX=2*MX))
        ((MY=2*MY))
        runcase $ST "-mx $MX -my $MY -mz 2" $REF
        ((MX=2*MX))
        ((MY=2*MY))
    done
done
