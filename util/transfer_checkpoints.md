Local -> Slurm
```bash
scp -r .\checkpoints\fine-wildflower-497-segformer-b2-231128-164424\ slurmmaster-ls6:/scratch/medfm/vesuv/kaggle1stReimp/checkpoints
```

Vast -> Local
```bash
.\util\get_checkpoint.sh fine-wildflower-497-segformer-b2-231128-164424 vast
```

Slurm -> Local
```bash
.\util\get_checkpoint.sh fine-wildflower-497-segformer-b2-231128-164424
```