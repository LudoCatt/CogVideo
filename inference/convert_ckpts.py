import argparse
import os
import shutil
import json
import torch
from safetensors.torch import save_file

# python convert_ckpts.py --sharded_bin_path /cluster/scratch/lcattaneo/outputs_full/ckpt_080_5steps/checkpoint-5/cogvideox_fp32.bin --output_dir /cluster/scratch/lcattaneo/Finetuned/CogVideoX-080-5steps

def main():
    """
    This script changes the checkpoint from the sharded .bin to the sharded .safetensors
    It is meant to be used after the following script:

    cd /cluster/scratch/lcattaneo/outputs_full/checkpoint-*

    python - <<'PY'
    import torch
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    ckpt_dir = "."  
    out_file = "cogvideox_fp32.bin" 

    convert_zero_checkpoint_to_fp32_state_dict(ckpt_dir, out_file)
    print(f"Wrote {out_file}")
    PY
    """
    parser = argparse.ArgumentParser(
        description="Emulate the manual conversion of a sharded .bin transformer checkpoint to SafeTensors."
    )
    parser.add_argument(
        "--sharded_bin_path",
        type=str,
        required=True,
        help="Path to the sharded .bin checkpoint directory created by the DeepSpeed conversion script e.g. /cluster/scratch/lcattaneo/outputs_full/checkpoint-700/cogvideox_fp32.bin "
             "This directory should contain 'pytorch_model-*.bin' files and a 'pytorch_model.bin.index.json' file.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default='/cluster/scratch/lcattaneo/CogVideoX'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/cluster/scratch/lcattaneo/CogVideoX-ft'
    )
    args = parser.parse_args()

    # 1. Setup the Destination Directory Structure
    print(f"Setting up the destination directory at '{args.output_dir}'...")
    transformer_output_path = os.path.join(args.output_dir, "transformer")
    os.makedirs(transformer_output_path, exist_ok=True)
    
    # Copy all other essential components from the base model first
    components_to_copy = ["scheduler", "text_encoder", "tokenizer", "vae"]
    for component in components_to_copy:
        src = os.path.join(args.base_model_path, component)
        dst = os.path.join(args.output_dir, component)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"   - Copied '{component}'")
            
    files_to_copy = ["configuration.json", "model_index.json"]
    for file_name in files_to_copy:
        src = os.path.join(args.base_model_path, file_name)
        dst = os.path.join(args.output_dir, file_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"   - Copied '{file_name}'")

    # Also copy config.json into the new transformers folder
    transformer_config_src = os.path.join(args.base_model_path, "transformer", "config.json")
    transformer_config_dst = os.path.join(transformer_output_path, "config.json")
    if os.path.isfile(transformer_config_src):
        shutil.copy2(transformer_config_src, transformer_config_dst)
        print(f"   - Copied 'config.json' to the transformer directory")
    else:
        print(f"   - WARNING: Transformer config.json not found at {transformer_config_src}")

    print("Destination structure is ready.\n")


    # 2. Convert Transformer Shards One-by-One
    print("Converting transformer shards from .bin to .safetensors...")
    
    # Load the original index file to know which files to convert
    original_index_path = os.path.join(args.sharded_bin_path, "pytorch_model.bin.index.json")
    if not os.path.isfile(original_index_path):
        raise FileNotFoundError(f"Source index file not found: {original_index_path}")
        
    with open(original_index_path, 'r') as f:
        original_index = json.load(f)
        
    # Get a unique, sorted list of the shard files
    shards_to_convert = sorted(list(set(original_index['weight_map'].values())))

    for shard_name in shards_to_convert:
        source_path = os.path.join(args.sharded_bin_path, shard_name)
        
        # This is where we emulate your renaming step.
        # e.g., pytorch_model-00001-of-00005.bin -> diffusion_pytorch_model-00001-of-00005.safetensors
        new_shard_name = shard_name.replace("pytorch_model", "diffusion_pytorch_model").replace(".bin", ".safetensors")
        destination_path = os.path.join(transformer_output_path, new_shard_name)
        
        print(f"   - Converting '{shard_name}' -> '{new_shard_name}'")
        
        # Load the PyTorch pickle (.bin) file
        state_dict = torch.load(source_path, map_location="cpu")
        
        # Save as a real SafeTensors file
        save_file(state_dict, destination_path)
    
    print("All shards converted.\n")


    # 3. Create the New, Correct JSON Index
    print("Creating new JSON index for the .safetensors model...")
    new_index = {"metadata": original_index.get("metadata", {}), "weight_map": {}}

    # Remap the weights to the new .safetensors filenames
    for tensor_name, old_shard_name in original_index['weight_map'].items():
        new_shard_name = old_shard_name.replace("pytorch_model", "diffusion_pytorch_model").replace(".bin", ".safetensors")
        new_index['weight_map'][tensor_name] = new_shard_name
        
    # Define the name for the new index file
    new_index_path = os.path.join(transformer_output_path, "diffusion_pytorch_model.safetensors.index.json")
    
    with open(new_index_path, 'w') as f:
        json.dump(new_index, f, indent=2)
        
    print(f"New index written to '{new_index_path}'\n")

    print(f"All done! Your complete, fine-tuned SafeTensors model is ready at:\n  {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()