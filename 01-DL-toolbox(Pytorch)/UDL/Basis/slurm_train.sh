#!/usr/bin/env bash

set -x

#cd projects/derain

PARTITION=defq
JOB_NAME=task
#CONFIG=$3
#WORK_DIR=$4
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
#NNODE=${NNODE:-'node[004]'}
#PY_ARGS=${@:5}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:0 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --nodelist=node[004-005] \
    ${SRUN_ARGS} \
    python -u main.py --launcher="slurm" #${PY_ARGS}

#srun -p defq -J test -n 2 --nodelist=node[004-005] --ntasks-per-node=2 --export=cuda_home python -u test_slurm.py
#srun --partition=defq --job-name=rain -n 1 --nodelist=node004 --gres=gpu:8 --ntasks-per-node=8 python -u derain_main.py --launcher="slurm
#sed -i "s/\r//" slurm_train.sh
# srun -p defq -J test -n 1 --nodelist=node[004] --ntasks-per-node=1 python -u derain_main.py --launcher slurm
#srun -p defq -J test -n 2 --nodelist=node[004-005] --ntasks-per-node=1 python -u test_slurm.py
