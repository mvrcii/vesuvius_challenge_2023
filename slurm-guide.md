
gpus = ['rtx4090', 'rtx3090', 'rtx2080ti']

```bash
sbatch -p ls6 --gres=gpu:rtx2080ti:8 --wrap="python pl_train.py" -o "slurm_logs/slurm-%j.out"
```