#!/bin/bash
#SBATCH -p tflmb_gpu-rtx4090 # partition (queue)
#SBATCH --mem 32000 # memory pool for all cores (20GB)
#SBATCH -c 16 # number of cores
#SBATCH -a 1-2 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /home/bratulic/git_repos/robo/cVLA # Change working_dir
#SBATCH -o /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

source ~/.bashrc

conda activate paligemma


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
    python hf_image_condition.py --no_augs --lr 1e-5 --p_copy 0.5 --save_steps 500 --extra_run_name _pcopy05_sorted025_noaugs --p_sort_by_l2_distance 0.25
    exit $?
fi

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
    python hf_image_condition.py --no_augs --lr 1e-5 --p_copy 0.25 --save_steps 500 --extra_run_name _pcopy025_sorted025_noaugs --p_sort_by_l2_distance 0.25
    exit $?
fi
