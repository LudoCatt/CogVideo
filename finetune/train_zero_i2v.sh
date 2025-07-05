#!/usr/bin/env bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=full_finetune
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:79g
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=100G
#SBATCH --output=logs/full_out.txt
#SBATCH --error=logs/full_err.txt

module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cogvideox

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "/cluster/scratch/lcattaneo/CogVideoX"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/cluster/scratch/lcattaneo/outputs_full"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/cluster/scratch/lcattaneo/data"
    --caption_column "prompts.txt"
    --video_column "videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 100 # number of training epochs
    --seed 42 # random seed

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/cluster/scratch/lcattaneo/outputs_full/checkpoint-1000"  
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

export TORCH_DATALOADER_PREFETCH_FACTOR=1

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
