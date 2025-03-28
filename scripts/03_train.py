# scripts/03_train.py
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import argparse
import os
import sys # Added for redirection
from contextlib import redirect_stdout, redirect_stderr # Added for redirection
import json
import time
import math # For potential use in ppl calculation if testing
from pathlib import Path
from mlx.utils import tree_flatten

# Import necessary mlx-lm components
try:
    # Core training loop and args object from tuner.trainer
    from mlx_lm.tuner.trainer import train, TrainingArgs, iterate_batches, default_loss, evaluate

    # Utility for loading model/tokenizer
    from mlx_lm.utils import load

    # Function to add LoRA layers
    from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule, print_trainable_parameters

    # Dataset loading function
    from mlx_lm.tuner.datasets import load_dataset

    print("Successfully imported components from tuner and utils.")

except ImportError as e:
    print(f"Failed to import required mlx-lm components. Check installation/version. Error: {e}")
    exit()

# Import Hugging Face datasets loader
from datasets import load_from_disk
# Import yaml for config file loading
import yaml

# --- Default Args - aligned with CLI examples/docs ---
DEFAULT_ARGS = {
    # Model and Data Paths
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "data": "./data/processed_dnd_data", # Directory containing train.jsonl/valid.jsonl
    "adapter_path": "./models/adapters", # Directory for saving adapters

    # LoRA Configuration
    "lora_layers": 16,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.0,
    "lora_scale": 20.0,

    # Training Hyperparameters
    "batch_size": 1, # Default, override via CLI or YAML
    "iters": 1000,
    "learning_rate": 1e-5,
    "max_seq_length": 1024,
    "seed": 42,

    # Reporting and Saving Intervals
    "steps_per_report": 10,
    "steps_per_eval": 100,
    "steps_per_save": 100, # Using the TrainingArgs name

    # Evaluation Control
    "test": False, # Run test set evaluation after training
    "val_batches": 25,
    "test_batches": 500, # Default for test eval

    # Hardware/Efficiency
    "grad_checkpoint": True,

    # Other Flags
    "train": True, # Set training mode
    "optimizer": "adam", # Optimizer choice
    "lr_schedule": None, # Optional learning rate schedule
    "fine_tune_type": "lora", # Type of tuning
    "resume_adapter_file": None, # Option to resume
    # Removed trust_remote_code from defaults
}
# --- End Default Args ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA with mlx-lm.")

    # Add arguments dynamically
    for key, value in DEFAULT_ARGS.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value, help=f"{key.capitalize()} (default: {value})")
        elif value is None or isinstance(value, (int, float, str)): # Handle None for lr_schedule etc.
             parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"{key.capitalize()} (default: {value})")
        # Add more specific handling here if needed (e.g., parsing dicts from CLI)

    # Add args specifically found in lora.py's parser not in defaults
    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides defaults, overridden by CLI args)")
    # Removed --trust_remote_code from argparse

    args = parser.parse_args()
    all_args_dict = vars(args) # Keep dict for saving config

    # YAML Config Loading (Optional)
    if args.config:
        print(f"Loading configuration from {args.config}")
        try:
            with open(args.config, "r") as file:
                yaml_config = yaml.safe_load(file)
                # Update only args that weren't explicitly set via CLI (which use default values)
                defaults_from_parser = {k: v for k, v in DEFAULT_ARGS.items() if k in all_args_dict}
                for key, value in yaml_config.items():
                     # Check if the current value matches the default AND the key is in YAML
                     # This means it wasn't set via CLI explicitly
                     if key in all_args_dict and all_args_dict[key] == defaults_from_parser.get(key):
                         all_args_dict[key] = value
                     # Or if the key wasn't even in the defaults/parser initially
                     elif key not in all_args_dict:
                         all_args_dict[key] = value
            args = argparse.Namespace(**all_args_dict) # Update args namespace
            print("YAML config loaded and applied.")
        except Exception as e:
            print(f"Error loading YAML config: {e}")

    # --- LOGGING SETUP ---
    log_dir = Path(args.data) # Use the data directory specified
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    log_file_path = log_dir / "training_run.log" # Use .log extension

    print(f"--- Starting Run ---") # Print before redirection
    print(f"Saving console output to: {log_file_path}")

    try:
        # Open the log file in write mode ('w') to overwrite previous logs
        # Change to 'a' to append if desired
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # Use context managers to redirect stdout and stderr to the file
            with redirect_stdout(log_file), redirect_stderr(log_file):

                # --- ALL THE REST OF YOUR MAIN FUNCTION GOES HERE ---
                print("Starting LoRA Fine-tuning...")
                print("Effective Arguments (inside log):")
                for key, value in vars(args).items():
                    print(f"  {key}: {value}")

                # Set Seed
                mx.random.seed(args.seed)

                # Define Paths (adapter path is separate)
                adapter_save_dir = Path(args.adapter_path)
                adapter_save_dir.mkdir(parents=True, exist_ok=True)
                adapter_checkpoint_file = adapter_save_dir / "adapters.safetensors" # Default name used by trainer

                # --- Load Model and Tokenizer ---
                print("\nLoading model and tokenizer...")
                try:
                    model, tokenizer = load(args.model)

                    # Handle Pad Token
                    if tokenizer.pad_token is None:
                         if tokenizer.eos_token:
                             print("Warning: Tokenizer missing pad token; using EOS token as pad token.")
                             tokenizer.pad_token = tokenizer.eos_token
                         else:
                             print("Error: Tokenizer missing pad token and EOS token. Cannot proceed.")
                             exit() # Exit cleanly inside the context if needed
                    print("Model and tokenizer loaded.")
                except Exception as e:
                    print(f"Error loading model/tokenizer: {e}")
                    import traceback
                    traceback.print_exc()
                    exit()

                # --- Load Datasets using the library function ---
                print("Loading datasets using load_dataset...")
                try:
                    # Pass the args namespace to load_dataset
                    train_dataset, valid_dataset, test_dataset = load_dataset(args, tokenizer)

                    # Validation checks
                    if args.train and train_dataset is None:
                         print(f"Error: Training enabled but no training data loaded from '{args.data}'. Check for 'train.jsonl'.")
                         exit()
                    if args.train and args.steps_per_eval > 0 and valid_dataset is None:
                         print(f"Warning: Evaluation enabled (steps_per_eval > 0) but no validation data loaded ('valid.jsonl' missing?). Evaluation will be skipped.")
                    if args.test and test_dataset is None:
                        print(f"Error: Testing enabled but no test data loaded ('test.jsonl' missing?).")
                        if not args.train: # Exit if only testing and no test data
                            exit()
                        # If training, can continue without test data, just skip test phase

                    print(f"Datasets loaded. Train dataset: {'OK' if train_dataset else 'Not Loaded'}, Valid dataset: {'OK' if valid_dataset else 'Not Loaded'}, Test dataset: {'OK' if test_dataset else 'Not Loaded'}")

                except Exception as e:
                    print(f"Error loading dataset(s) from '{args.data}': {e}")
                    import traceback
                    traceback.print_exc()
                    exit()

                # --- Prepare Model for Training (Freeze & Apply LoRA) ---
                if args.train: # Only modify model if training
                    print("\nPreparing model for training...")
                    model.freeze()
                    try:
                        # Structure LoRA params using names from args
                        lora_params_dict = {
                            "rank": args.lora_rank,
                            "alpha": args.lora_alpha,
                            "dropout": args.lora_dropout,
                            "scale": args.lora_scale,
                        }

                        # Handle fine_tune_type logic as in lora.py's train_model
                        if args.fine_tune_type == "full":
                             print("Configuring for full fine-tuning (unfreezing layers)...")
                             num_layers_to_unfreeze = getattr(args, 'num_layers', -1) # Default to all if not specified
                             if num_layers_to_unfreeze <= 0:
                                  model.unfreeze()
                             else:
                                 if num_layers_to_unfreeze > len(model.layers):
                                     print(f"Warning: Requested to unfreeze {num_layers_to_unfreeze} layers but model only has {len(model.layers)}. Unfreezing all.")
                                     model.unfreeze()
                                 else:
                                     for l in model.layers[-num_layers_to_unfreeze:]:
                                         l.unfreeze()
                        elif args.fine_tune_type in ["lora", "dora"]:
                             print(f"Applying {args.fine_tune_type.upper()} layers...")
                             linear_to_lora_layers(
                                 model=model,
                                 num_layers=args.lora_layers, # Use lora_layers arg here
                                 config=lora_params_dict,      # Pass the structured dict
                                 use_dora=(args.fine_tune_type == "dora")
                             )
                             print(f"Applied {args.fine_tune_type.upper()} with config: {lora_params_dict} to {args.lora_layers} layers.")
                        else:
                             print(f"Error: Unknown fine_tune_type '{args.fine_tune_type}'")
                             exit()

                        model.train() # Set model to training mode

                        # Parameter counting using utility function
                        print("\nCalculating parameter counts...")
                        print_trainable_parameters(model)

                        # Resume from checkpoint if specified
                        if args.resume_adapter_file:
                            print(f"Resuming from adapter file: {args.resume_adapter_file}")
                            # load_weights is safer as it handles missing/extra keys
                            model.load_weights(args.resume_adapter_file, strict=False)

                    except Exception as e:
                        print(f"Error preparing model for training: {e}")
                        import traceback
                        traceback.print_exc()
                        exit()

                    # --- Create Optimizer ---
                    print("\nCreating optimizer...")
                    try:
                        # Build learning rate schedule if specified in args.lr_schedule (e.g. via YAML)
                        lr_schedule_config = getattr(args, 'lr_schedule', None)
                        lr = build_schedule(lr_schedule_config) if lr_schedule_config else args.learning_rate

                        # Add support for AdamW with weight decay (common)
                        weight_decay = getattr(args, 'weight_decay', 0.0) # Add weight_decay to defaults or YAML if needed

                        if args.optimizer.lower() == "adam":
                            optimizer = optim.Adam(learning_rate=lr)
                        elif args.optimizer.lower() == "adamw":
                             optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
                        # Add other optimizers (SGD, Adafactor) if needed
                        else:
                            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
                        print(f"Optimizer created: {args.optimizer} with effective LR and weight_decay={weight_decay}.")
                    except Exception as e:
                         print(f"Error creating optimizer: {e}")
                         exit()

                    # --- Create TrainingArgs ---
                    # Pass save_every, eval_every directly
                    training_args_obj = TrainingArgs(
                        batch_size=args.batch_size,
                        iters=args.iters,
                        val_batches=args.val_batches,
                        steps_per_report=args.steps_per_report,
                        steps_per_eval=args.steps_per_eval if valid_dataset else -1, # Check if val data exists
                        steps_per_save=args.steps_per_save, # Explicitly use this
                        adapter_path=str(adapter_save_dir), # Pass dir path, trainer handles file name now
                        max_seq_length=args.max_seq_length,
                        grad_checkpoint=args.grad_checkpoint,
                    )

                    print("\nTrainingArgs (controls loop):")
                    print(f"  iters: {training_args_obj.iters}")
                    print(f"  batch_size: {training_args_obj.batch_size}")
                    print(f"  save_every: {training_args_obj.steps_per_save}")
                    print(f"  eval_every: {training_args_obj.steps_per_eval}")

                    # --- Start Training ---
                    print("\nInitiating training (calling tuner.trainer.train)...")
                    start_time = time.time()
                    try:
                        # Call the low-level train function from tuner.trainer
                        train(
                            model=model,                 # Prepared model
                            tokenizer=tokenizer,         # Loaded tokenizer
                            optimizer=optimizer,         # Created optimizer
                            train_dataset=train_dataset, # Loaded dataset object
                            val_dataset=valid_dataset,   # Loaded dataset object
                            args=training_args_obj,      # Training loop controls
                            loss=default_loss,           # Default loss fn
                            iterate_batches=iterate_batches # Default batch iterator
                        )
                    except Exception as e:
                        print(f"\nAn unexpected error occurred during training: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue to saving config if possible, maybe don't exit

                    end_time = time.time()
                    print(f"\nTraining finished in {(end_time - start_time):.2f} seconds!")
                    print(f"LoRA adapter checkpoints saved to directory: {adapter_save_dir}")
                    # Final weights are usually saved automatically by train() to adapter_path/adapters.safetensors


                # --- Optional Test Set Evaluation ---
                if args.test:
                    if test_dataset is None:
                        print("\nSkipping test evaluation as test data was not loaded.")
                    else:
                        print("\nEvaluating on test set...")
                        start_time = time.time()
                        model.eval() # Set model to eval mode

                        # Define path to final adapters more robustly
                        final_adapter_file = adapter_save_dir / "adapters.safetensors"

                        # Load final trained adapters if training just occurred and file exists
                        if args.train and final_adapter_file.is_file():
                            print(f"Loading final adapters from {final_adapter_file} for test evaluation.")
                            model.load_weights(str(final_adapter_file), strict=False)
                        # If *only* testing, need to load base model then apply adapters manually or load merged
                        elif not args.train and args.adapter_path:
                            adapter_file_to_test = Path(args.adapter_path) / "adapters.safetensors"
                            if adapter_file_to_test.is_file():
                                print(f"Loading adapters from {adapter_file_to_test} for test evaluation.")
                                # Assume model loaded earlier is the base model
                                model.load_weights(str(adapter_file_to_test), strict=False)
                            else:
                                print(f"Warning: Testing enabled, but no adapter file found at {adapter_file_to_test}. Evaluating base model or previously loaded state.")
                        elif not args.train:
                             print("Warning: Testing enabled but no adapter path specified. Evaluating base model.")


                        test_loss = evaluate(
                            model=model,
                            dataset=test_dataset,
                            tokenizer=tokenizer,
                            batch_size=args.batch_size, # Can use larger batch size for eval if memory allows
                            num_batches=args.test_batches,
                            max_seq_length=args.max_seq_length, # Use same seq length
                            loss=default_loss,
                            iterate_batches=iterate_batches
                        )
                        test_ppl = math.exp(test_loss) if test_loss is not None else float('inf')
                        end_time = time.time()
                        print(f"Test Loss: {test_loss:.3f}, Test Perplexity: {test_ppl:.3f}")
                        print(f"Test evaluation finished in {(end_time - start_time):.2f} seconds.")


                # Save the final configuration used to run this script
                final_config_path = adapter_save_dir / "training_run_config.json"
                serializable_args = {k: v for k, v in all_args_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))} # Include None
                try:
                    with open(final_config_path, 'w') as f:
                        json.dump(serializable_args, f, indent=4)
                    print(f"\nTraining run configuration saved to: {final_config_path}")
                except Exception as e:
                     print(f"Could not save training run config: {e}")

                # --- END OF CODE WITHIN REDIRECTION CONTEXT ---

        print(f"--- Run Finished ---") # Print after redirection ends
        print(f"Full log saved to: {log_file_path}")

    except Exception as e:
        # If an error happens during file setup or the main block
        print(f"\n!!! An error occurred outside or during logging setup: {e}", file=sys.__stderr__) # Print error to original stderr
        import traceback
        traceback.print_exc(file=sys.__stderr__) # Print traceback to original stderr
        # Also try logging the error to the file if it was opened
        try:
            with open(log_file_path, 'a', encoding='utf-8') as error_log:
                 print(f"\n!!! An error occurred: {e}", file=error_log)
                 traceback.print_exc(file=error_log)
        except:
            pass # Ignore errors during error logging

if __name__ == "__main__":
    main()