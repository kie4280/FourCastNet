# How to run

## For running single GPU per node and multiple node in batch mode (non-interactive)

```
sbatch ddp/sbatch_run.sh
```


## For running single GPU per node and multiple node in interactive mode (using srun)

```
srun --nodes=NUMBER_OF_NODES --ntasks-per-node=NUMBER_OF_TASKS_PER_NODE ddp/srun_run.sh
```

## Some notes

`torchrun` automatically sets `MASTER_ADDR` and `MASTER_PORT` to the FQDN of the head node. 
If the FQDN is not setup correctly on all machines, the job will fail (*error: failed to resolve ipv6 address...*).