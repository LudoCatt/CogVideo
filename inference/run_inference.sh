#!/usr/bin/env bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=u_inf
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=50G
#SBATCH --output=logs/u_inf_out.txt
#SBATCH --error=logs/u_inf_err.txt

# modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cogvideox

python /cluster/home/lcattaneo/CogVideo/inference/run_inference.py \
    --script_dir   /cluster/home/lcattaneo/CogVideo/inference \
    --prompt_file  /cluster/scratch/lcattaneo/data_exp/data_inf/inf_prompt.txt \
    --frame_dir    /cluster/scratch/lcattaneo/data_exp/data_inf/inference \
    --output_dir   /cluster/scratch/lcattaneo/i2v_out/Experiment \
    --model_path   /cluster/scratch/lcattaneo/CogVideoX \
#    --lora_path    /cluster/scratch/lcattaneo/outputs/checkpoint-2250/pytorch_lora_weights.safetensors


python cli_demo.py --prompt "A sleek, yellow-and-black quadruped robot stands poised on a tiled, grayscale plane under soft studio lighting. With a confident gait, it begins walking toward the right side of the screen, each of its articulated legs moving in smooth, deliberate coordination. The front right leg lifts first, followed by the rhythmic advance of the others, creating a steady trotting motion. The robot maintains perfect balance, its compact body staying level while subtle joint adjustments in its legs absorb virtual ground impacts. The camera remains steadily fixed on the robot’s side, capturing the mechanical elegance and purposeful stride of its movement. Shadows shift naturally beneath it, emphasizing both realism and directional momentum. The ambient environment stays minimal and unchanging, keeping full visual attention on the robot’s continuous, self-assured march." \
    --image_or_video_path "/cluster/scratch/lcattaneo/data_exp2/data_100/first_frames/demo_seed_0.png" \
    --model_path "/cluster/scratch/lcattaneo/Finetuned/CogVideoX-100-5steps" \
    --output_path "/cluster/scratch/lcattaneo/i2v_out/Experiment2/G1_100_5steps.mp4" \
    --num_frames 49 \
    --width 720 \
    --height 480 \
    --generate_type "i2v" \
    --fps 8