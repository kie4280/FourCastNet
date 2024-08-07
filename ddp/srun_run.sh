#!/bin/bash

echo "nodelist: $SLURM_JOB_NODELIST"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "head node: $head_node"
head_node_ip=($(getent ahosts "$head_node" | tail -n 1 | cut -d ' ' -f 1))

echo head node IP: $head_node_ip
export LOGLEVEL=INFO
export MASTER_PORT=23458
export MASTER_ADDR_OVERRIDE="$head_node_ip"
export RUN_ID=38567

torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node $SLURM_NTASKS_PER_NODE \
--rdzv_id $RUN_ID \
--rdzv_endpoint "$head_node_ip:$MASTER_PORT" \
--rdzv_backend c10d \
~/FourCastNet/ddp/torchrun_example.py



# --master_addr "$MASTER_ADDR" \
# --master_port "$MASTER_PORT" \