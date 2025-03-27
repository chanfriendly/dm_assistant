# scripts/04_fuse_and_quantize.py
import mlx.core as mx
# Ensure these imports are correct for your mlx-lm version
try:
    from mlx_lm.utils import load
    from mlx_lm.quantize import quantize
except ImportError as e:
     print(f"Failed to import mlx_lm components. Check installation. Error: {e}")
     exit()

import argparse
import os
import json
from pathlib import Path

# --- Configuration ---
DEFAULT_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct" # MUST match the training base model
DEFAULT_ADAPTER_PATH = "../models/adapters/"          # Path to saved LoRA adapters
DEFAULT_OUTPUT_DIR = "../models/dnd_expert_quantized" # Save quantized model here
DEFAULT_QUANTIZE_BITS = 4                             # 4-bit is good for M1
DEFAULT_GROUP_SIZE = 64
# ---

def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters and quantize the model.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Path or HF ID of the base model used for training.")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH, help="Path to the saved LoRA adapters directory.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the fused and quantized model.")
    parser.add_argument("--q_bits", type=int, default=DEFAULT_QUANTIZE_BITS, help="Bits for quantization (e.g., 4, 8).")
    parser.add_argument("--group_size", type=int, default=DEFAULT_GROUP_SIZE, help="Group size for quantization.")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True, help="Trust remote code (e.g., for Phi-3).")

    args = parser.parse_args()

    print("Starting model fusion and quantization...")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Quantization Bits: {args.q_bits}")
    print(f"Group Size: {args.group_size}")

    adapter_file = Path(args.adapter_path) / "adapters.safetensors"
    config_file = Path(args.adapter_path) / "adapter_config.json" # Or sometimes lora_config.json

    if not adapter_file.is_file():
         adapter_file_npz = Path(args.adapter_path) / "adapters.npz" # Check for older format too
         if adapter_file_npz.is_file():
             adapter_file = adapter_file_npz # Use NPZ if found
         else:
             print(f"Error: Adapter file (adapters.safetensors or adapters.npz) not found in {args.adapter_path}")
             return
             
    if not config_file.is_file():
         # Try alternate common name
         config_file_alt = Path(args.adapter_path) / "config.json" 
         if config_file_alt.is_file():
             config_file = config_file_alt
         else:
            print(f"Warning: Adapter config file (adapter_config.json or config.json) not found in {args.adapter_path}. Fusion might use defaults.")
            # Fusion might still work if load can infer params, but it's less reliable.

    # --- Load Model with Adapters (mlx-lm load handles fusion) ---
    print("\nLoading base model and fusing adapters...")
    try:
        model, tokenizer = load(
            args.base_model,
            adapter_path=args.adapter_path, # Pass the directory
            trust_remote_code=args.trust_remote_code
        )
        print("Model loaded and adapters fused successfully.")
    except Exception as e:
        print(f"Error loading model and adapters: {e}")
        print("Check if the base model identifier and adapter path are correct.")
        return

    # --- Quantize the Fused Model ---
    print(f"\nQuantizing the fused model to {args.q_bits}-bit...")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        # quantize function returns the path to the quantized weights
        quantized_weights_path = quantize(
            model=model,          # Fused model
            tokenizer=tokenizer,  # Pass tokenizer to save it alongside
            bits=args.q_bits,
            group_size=args.group_size,
            output_dir=Path(args.output_dir) # quantize expects a Path object for output dir
        )
        print(f"Quantization complete.")
        print(f"Quantized model weights saved in: {quantized_weights_path}")
        print(f"Full quantized model (including config/tokenizer) is in: {args.output_dir}")

        # Save quantization config if quantize doesn't do it automatically
        # (It usually saves a config.json with quantization details now)
        q_config_path = Path(args.output_dir) / "quantization.json" # Or check config.json
        if not (Path(args.output_dir) / "config.json").exists() or "quantization" not in json.load(open(Path(args.output_dir) / "config.json")):
             quant_config = {"quantization": {"bits": args.q_bits, "group_size": args.group_size}}
             try:
                with open(q_config_path, 'w') as f:
                     json.dump(quant_config, f, indent=4)
                print(f"Quantization config saved to {q_config_path}")
             except Exception as e:
                  print(f"Could not save quantization config: {e}")


    except Exception as e:
        import traceback
        print(f"Error during quantization: {e}")
        traceback.print_exc() # Print stack trace for detailed error
        return

    print("\nFusion and Quantization process finished.")

if __name__ == "__main__":
    main()