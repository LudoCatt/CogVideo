#!/usr/bin/env bash
#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=24G
#SBATCH --output=logs/inference_out.txt
#SBATCH --error=logs/inference_err.txt

# modules
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cogvideox

# paths
SCRIPT_DIR="/cluster/home/lcattaneo/CogVideo/inference"
MODEL_PATH="/cluster/scratch/lcattaneo/CogVideoX"
LORA_PATH="/cluster/scratch/lcattaneo/outputs/checkpoint-2250/pytorch_lora_weights.safetensors" 
PROMPT_FILE="/cluster/scratch/lcattaneo/data/prompts.txt"
FRAME_DIR="/cluster/scratch/lcattaneo/images"
OUT_DIR="/cluster/scratch/lcattaneo/i2v_out/LoRA"

mkdir -p "$OUT_DIR"
mapfile -t PROMPTS < "$PROMPT_FILE"

# main loop
#         --lora_path "" \
for i in "${!PROMPTS[@]}"; do
    IMG=$(ls "$FRAME_DIR" | sort | sed -n "$((i+1))p") || continue
    [[ -z "$IMG" || -z "${PROMPTS[$i]}" ]] && continue

    python "$SCRIPT_DIR/cli_demo.py" \
        --prompt "${PROMPTS[$i]}" \
        --image_or_video_path "$FRAME_DIR/$IMG" \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --output_path "$OUT_DIR/$(printf "%04d.mp4" "$i")" \
        --num_frames 49 \
        --width 720 --height 480 \
        --generate_type i2v
done

echo "All videos saved in $OUT_DIR"
