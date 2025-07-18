#!/usr/bin/env bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=MuJoCo_t_inf
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=50G
#SBATCH --output=logs/MuJoCo_t_inf_out.txt
#SBATCH --error=logs/MuJoCo_t_inf_err.txt

# modules
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cogvideox

python /cluster/home/lcattaneo/CogVideo/inference/run_inference.py \
    --script_dir   /cluster/home/lcattaneo/CogVideo/inference \
    --prompt_file  /cluster/scratch/lcattaneo/inf_data/inf_prompts.txt \
    --frame_dir    /cluster/scratch/lcattaneo/inf_data/first_frames \
    --output_dir   /cluster/scratch/lcattaneo/i2v_out/MuJoCo_tuned_inf \
    --model_path   /cluster/scratch/lcattaneo/CogVideoX-ft \
    --lora_path    /cluster/scratch/lcattaneo/outputs/checkpoint-2250/pytorch_lora_weights.safetensors