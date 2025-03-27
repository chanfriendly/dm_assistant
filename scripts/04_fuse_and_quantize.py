# scripts/04_fuse_and_quantize.py
import mlx.core as mx
import argparse
import os
import json
import shutil # For copying files
from pathlib import Path
from mlx.utils import tree_flatten
import glob # Make sure glob is imported

# Import necessary mlx-lm components
try:
    from mlx_lm.utils import (
        load,
        quantize_model,
        save_weights,
        save_config,
        get_model_path
        )
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    # Import remove_lora_layers
    from mlx_lm.tuner.utils import remove_lora_layers

    print("Successfully imported components from mlx_lm.utils and tuner.utils.")

except ImportError as e:
    print(f"Failed to import required mlx-lm components. Check installation/version. Error: {e}")
    exit()

# Import transformers only if needed for config fallback
try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None
    print("Warning: transformers library not found. Config loading fallback might fail.")


# --- Configuration ---
DEFAULT_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_ADAPTER_PATH = "./models/adapters/"
DEFAULT_OUTPUT_DIR = "./models/dnd_expert_quantized"
DEFAULT_QUANTIZE_BITS = 4
DEFAULT_GROUP_SIZE = 64
DEFAULT_DTYPE = "float16"
# ---

def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters, remove LoRA layers, and quantize the model.")
    # Argument parsing
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="HF ID of the base model.")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH, help="Path to the saved LoRA adapters directory.")
    parser.add_argument("--output-path", "--mlx-path", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the final model.")
    parser.add_argument("-q", "--quantize", action='store_true', default=True, help="Apply quantization.")
    parser.add_argument("--no-quantize", dest="quantize", action='store_false', help="Do not quantize.")
    parser.add_argument("--q-bits", type=int, default=DEFAULT_QUANTIZE_BITS, help="Bits for quantization.")
    parser.add_argument("--q-group-size", type=int, default=DEFAULT_GROUP_SIZE, help="Group size for quantization.")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="Dtype for non-quantized weights.")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True, help="Trust remote code for config loading.")

    args = parser.parse_args()

    # Initial prints and path validation
    print("Starting model fusion, LoRA removal, and quantization...")
    print(f"Base Model: {args.model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Output Path: {args.output_path}")
    if args.quantize:
        print(f"Quantizing to {args.q_bits}-bit with group size {args.q_group_size}.")
    print(f"Saving non-quantized weights as: {args.dtype}")

    output_path = Path(args.output_path)
    adapter_path = Path(args.adapter_path)

    # Validate paths
    if not adapter_path.is_dir():
         print(f"Error: Adapter path '{args.adapter_path}' is not a directory.")
         exit()
    if not (adapter_path / "adapters.safetensors").is_file():
         print(f"Error: Adapter file 'adapters.safetensors' not found in '{args.adapter_path}'.")
         exit()
    # Ensure adapter_config.json exists (we created it manually)
    if not (adapter_path / "adapter_config.json").is_file():
          print(f"Error: Config file 'adapter_config.json' not found in '{args.adapter_path}'. Please create it.")
          exit()

    if output_path.exists() and not output_path.is_dir():
        print(f"Error: Output path '{args.output_path}' exists and is not a directory.")
        exit()
    elif output_path.exists() and any(output_path.iterdir()):
         print(f"Warning: Output directory '{args.output_path}' already exists and is not empty. Files might be overwritten.")
    output_path.mkdir(parents=True, exist_ok=True)


    # --- Load Model (Fusing Adapters) ---
    print("\nLoading base model and fusing adapters...")
    config = None
    model = None
    tokenizer = None
    try:
        # Load should fuse based on adapter_path
        model, tokenizer = load(
            path_or_hf_repo=args.model,
            adapter_path=str(adapter_path)
            # Removed trust_remote_code
        )
        print("Model loaded and adapters fused successfully.")

        # Retrieve config
        if hasattr(model, 'config'):
             config = model.config
             if hasattr(config, 'to_dict') and callable(config.to_dict):
                 config = config.to_dict()
             elif not isinstance(config, dict):
                  config = None
        if config is None and AutoConfig: # Fallback
            print("Attempting to load base model configuration separately...")
            try:
                 hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
                 config = hf_config.to_dict()
                 print("Loaded base model config separately.")
            except Exception as config_e:
                 print(f"Warning: Could not load base config: {config_e}")

        if not isinstance(config, dict):
             print("Error: Failed to obtain model configuration dictionary.")
             exit()

    except Exception as e:
        print(f"Error loading model/tokenizer/fusing: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Explicitly Remove LoRA Layers ---
    print("\nRemoving LoRA layer structures from the fused model...")
    try:
        model = remove_lora_layers(model)
        print("LoRA layer structures removed.")
        # Verification print
        print("Verifying parameter counts after LoRA removal...")
        params_flat = tree_flatten(model.parameters())
        p_sum = sum(p.size for _, p in params_flat if isinstance(p, mx.array))
        trainable_params_flat = tree_flatten(model.trainable_parameters())
        t_sum = sum(p.size for _, p in trainable_params_flat if isinstance(p, mx.array))
        print(f"Total parameters: {p_sum:,}")
        print(f"Trainable parameters: {t_sum:,}")

    except Exception as e:
        print(f"Error removing LoRA layers: {e}")
        import traceback
        traceback.print_exc()
        # Continue cautiously

    # --- Quantize the Fused Model ---
    # Get weights AFTER fusion AND LoRA removal
    weights = dict(tree_flatten(model.parameters()))

    if args.quantize:
        print("\nQuantizing the fused model...")
        try:
            weights, config = quantize_model(
                model,
                config, # Pass the config dictionary
                args.q_group_size,
                args.q_bits,
            )
            print(f"Quantization to {args.q_bits}-bit complete.")
            # Ensure quantization info is in config
            if "quantization" not in config:
                 config["quantization"] = {"group_size": args.q_group_size, "bits": args.q_bits}
                 print("Manually added quantization info to config dict.")

        except Exception as e:
            print(f"Error during quantization: {e}")
            import traceback
            traceback.print_exc()
            exit()
    else:
        print("\nSkipping quantization.")
        try:
             save_dtype = getattr(mx, args.dtype)
             weights = {k: v.astype(save_dtype) for k, v in weights.items()}
        except AttributeError:
             print(f"Error: Invalid dtype '{args.dtype}' specified. Using float16.")
             weights = {k: v.astype(mx.float16) for k, v in weights.items()}

# *** CORRECTED INDENTATION FOR SAVING SECTION ***
    # --- Save Final Weights ---
    print(f"\nSaving weights to {output_path}...")
    try:
        save_weights(output_path, weights, donate_weights=True)
        print("Weights saved.")
    except Exception as e:
        print(f"Error saving weights: {e}")
        exit()

    # --- Save Config ---
    print("Saving config file...")
    try:
        save_config(config, config_path=output_path / "config.json")
        print("Config saved.")
    except Exception as e:
        print(f"Error saving config: {e}")

    # --- Save Tokenizer ---
    print("Saving tokenizer files...")
    try:
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)
            print("Tokenizer saved.")
        else:
            print("Error: Tokenizer object not loaded correctly, cannot save.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")

    # --- Copy Model Python Files ---
    print("Copying model python files (if any)...")
    try:
        source_model_path = get_model_path(args.model)
        py_files = glob.glob(str(source_model_path / "*.py"))
        copied_files = 0
        for file in py_files:
            try:
                target_file = output_path / os.path.basename(file)
                if not target_file.exists():
                    shutil.copy(file, output_path)
                    copied_files += 1
            except shutil.SameFileError:
                 pass
            except Exception as copy_e:
                 print(f"  Warning: Could not copy file {os.path.basename(file)}: {copy_e}")
        if copied_files > 0:
            print(f"Copied {copied_files} python files.")
        else:
            print("No python files found or copied.")
    except Exception as e:
        print(f"Error finding or copying python files: {e}")

    print(f"\nFusion, removal, and quantization process finished. Final model saved in: {args.output_path}")

if __name__ == "__main__":
    main()