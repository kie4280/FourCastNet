#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

echo "nodelist: $SLURM_JOB_NODELIST"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=($(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address))
head_node_ip=${head_node_ip[1]}

echo head node IP: $head_node_ip
export LOGLEVEL=INFO
export MASTER_PORT=12345
export MASTER_ADDR="$head_node_ip"

srun torchrun \
--nnodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
ddp/torchrun_example.py



# --rdzv_endpoint $head_node_ip:$MASTER_PORT \
# --master_addr "$MASTER_ADDR" \
# --master_port "$MASTER_PORT" \