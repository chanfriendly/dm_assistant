# scripts/03_train.py
import mlx.core as mx
import mlx.optimizers as optim
import argparse
import os
import json
import time
from pathlib import Path

# Import necessary mlx-lm components
try:
    # Use the train function directly from lora (common in newer versions)
    from mlx_lm.lora import train

    # Utility for loading model/tokenizer
    from mlx_lm.utils import load

    # TrainingArgs class location
    from mlx_lm.lora import TrainingArgs

    # Batch iteration function
    from mlx_lm.lora import iterate_batches

    # Default loss function
    from mlx_lm.lora import default_loss

except ImportError as e:
    print(f"Failed to import required mlx-lm components from 'lora' module. Check installation/version. Error: {e}")
    print("Attempting imports from tuner/trainer...")
    try:
        from mlx_lm.tuner.trainer import train
        from mlx_lm.utils import load # Usually stays in utils
        from mlx_lm.tuner.trainer import TrainingArgs
        from mlx_lm.tuner.trainer import iterate_batches
        from mlx_lm.tuner.trainer import default_loss
    except ImportError as e2:
        print(f"Also failed to import from tuner/trainer. Critical import error. Error: {e2}")
        exit()


from datasets import load_from_disk

# --- Default Args --- (Keep as before, removed trust_remote_code as it caused issues with load)
DEFAULT_ARGS = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "data_path": "./data/processed_dnd_data",
    "adapter_path": "./models/adapters",
    "lora_layers": 16,
    "lora_rank": 8,
    "lora_alpha": 16,
    "batch_size": 1,
    "iters": 1000,
    "learning_rate": 1e-5,
    "max_seq_length": 1024,
    "seed": 42,
    "steps_per_report": 10,
    "steps_per_eval": 100,
    "steps_per_save": 100,
    "test": False,
    "val_batches": 25,
    "grad_checkpoint": True,
    # Add trust_remote_code back here if needed by train() itself
    "trust_remote_code": True
}
# --- End Default Args ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA with mlx-lm.")

    # Add arguments dynamically from DEFAULT_ARGS
    for key, value in DEFAULT_ARGS.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value, help=f"{key.capitalize()} (default: {value})")
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"{key.capitalize()} (default: {value})")

    args = parser.parse_args()
    all_args = vars(args)

    print("Starting LoRA Fine-tuning...")
    print("Arguments:")
    for key, value in all_args.items():
        print(f"  {key}: {value}")

    # Set Seed
    mx.random.seed(args.seed)

    # Ensure adapter path exists
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"

    # --- Load Model and Tokenizer ---
    print("\nLoading model and tokenizer...")
    try:
        # Load without trust_remote_code as it failed before
        model, tokenizer = load(args.model)
        print("Model and tokenizer loaded.")
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Load Datasets ---
    print("Loading datasets...")
    try:
        dataset = load_from_disk(args.data_path)
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset.get("test"))
        if val_dataset is None:
            print("Warning: No 'validation' or 'test' split found. Evaluation during training will be skipped.")
            args.test = False
        print(f"Datasets loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset) if val_dataset else 'N/A'}")
    except Exception as e:
        print(f"Error loading dataset from {args.data_path}: {e}")
        exit()

    # --- Prepare Training Args ---
    # Create TrainingArgs instance with parameters it expects
    training_args_obj = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval if args.test and val_dataset else -1, # Disable if no test data or test=False
        steps_per_save=args.steps_per_save,
        max_seq_length=args.max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=args.grad_checkpoint,
    )

    print("\nTrainingArgs:")
    # Print key TrainingArgs details
    print(f"  batch_size: {training_args_obj.batch_size}")
    print(f"  iters: {training_args_obj.iters}")
    print(f"  max_seq_length: {training_args_obj.max_seq_length}")
    print(f"  adapter_file: {training_args_obj.adapter_file}")

    print("\nInitiating training...")
    start_time = time.time()
    try:
        # Call train, passing LoRA params and other relevant args directly
        # It's likely the top-level train function handles model prep internally
        train(
            model=model, # Pass the LOADED model object
            tokenizer=tokenizer, # Pass the LOADED tokenizer object
            optimizer=optim.Adam(learning_rate=args.learning_rate),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args_obj, # Still pass TrainingArgs for loop control
            loss=default_loss,
            iterate_batches=iterate_batches,
            # Pass LoRA config AGAIN - maybe needed alongside loaded model?
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            # test=args.test # Maybe test is part of TrainingArgs? Omit if so.
            # seed=args.seed # Seed is set globally, maybe train doesn't need it.
        )
    except TypeError as e:
        print(f"\nTypeError during train call: {e}")
        print("This likely means the arguments passed directly (like lora_layers, model path, test, seed)")
        print("are not expected by this specific `train` function.")
        print("Try passing the loaded `model` and `tokenizer` objects instead of the model path string.")
        print("Also verify if `trust_remote_code` is needed by `train`.")
        import traceback
        traceback.print_exc()
        exit()
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        exit()

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds!")
    print(f"LoRA adapters saved in: {adapter_path}")

    # Save the configuration used
    final_config_path = adapter_path / "training_config.json"
    serializable_args = {k: v for k, v in all_args.items() if isinstance(v, (str, int, float, bool, list, dict))}
    try:
        with open(final_config_path, 'w') as f:
            json.dump(serializable_args, f, indent=4)
        print(f"Training configuration saved to: {final_config_path}")
    except Exception as e:
         print(f"Could not save training config: {e}")

if __name__ == "__main__":
    main()