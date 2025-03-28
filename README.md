# D&D 5e Rules Expert - MLX Fine-Tuning Project

This project aims to fine-tune a lightweight, open-source large language model (LLM) to become an expert on Dungeons & Dragons 5th Edition rules. The primary focus is on leveraging Apple Silicon hardware using MLX for efficient training and inference.

The goal is to create a helpful assistant for DMs and players that can quickly answer specific rules questions based on the official sourcebooks.

**Current Status (As of [Insert Current Date, e.g., 2024-03-28]):**

*   Initial project pipeline setup (data generation, preparation, training, quantization, evaluation, inference scripts) is complete.
*   Utility scripts added:
    *   `visualize_log.py`: Parses training logs to plot loss curves for analysis.
    *   `review_data.py`: Helps inspect and review samples from the training/validation data files.
*   Data generation (`01_generate_qa.py`) using Ollama (`llama3.2`) and Markdown source files is functional.
*   Data preparation (`02_prepare_data.py`) into JSONL format is functional.
*   Training dataset **increased** to approximately **3700 Q&A pairs**. Manual curation is ongoing.
*   Training script (`03_train.py`) updated to automatically log output to a file (`training_run.log` in the data directory).
*   Recent training run completed using LoRA (rank 8, alpha 16) on `microsoft/Phi-3-mini-4k-instruct` with the ~3700 pair dataset and a batch size of 4 for 1000 iterations.
    *   Training loss stabilized compared to earlier runs (batch size 1).
    *   Validation loss plateaued around ~1.1, with the best checkpoint appearing around iteration 500. Log visualization confirmed this trend.
*   Fusion and quantization script (`04_fuse_and_quantize.py`) successfully processed the adapters from the latest training run (best checkpoint) and the base model into a quantized MLX model.
*   Evaluation script (`05_evaluate.py`) successfully loaded the *quantized* model and ran inference on a test set.
*   **Evaluation Results Analysis:** Manual review of the **latest quantized model's output** (`evaluation_results/evaluation_outputs.jsonl`) revealed **significant and frequent issues (Rated: F)**, including:
    *   Major **hallucinations** (inventing incorrect lore, rules, sources).
    *   Numerous **rule inaccuracies** on specific mechanics.
    *   Incomplete answers and generation artifacts.
    *   **Conclusion:** The model in its current *quantized* state is **unreliable and not suitable** for use as a rules expert.
*   **Next Steps:**
    1.  **Data Quality Focus:** Continue rigorous review and curation of the ~3700 Q&A pairs using `review_data.py` to ensure accuracy, conciseness, and relevance (removing potential noise like non-core D&D concepts if applicable).
    2.  **Evaluate Non-Quantized Model:** Test the *non-quantized* fused model (from the best training checkpoint, e.g., Iter 500) to determine if quantization significantly degraded performance.
    3.  **Iterate on Training:** Based on evaluation results, potentially re-train with the curated dataset, possibly adjusting hyperparameters (e.g., adding a learning rate scheduler, increasing iterations with early stopping, experimenting cautiously with LoRA rank/alpha).
    4.  **Root Cause Analysis:** If issues persist even with curated data and a non-quantized model, further investigation into the base model's suitability, prompt formatting, or fine-tuning process may be needed.
 
![loss_curve](https://github.com/user-attachments/assets/6395dd35-9fbb-4db9-a08b-353883f673a7)

## Features

*   Uses **MLX** for hardware-accelerated training and inference on Apple Silicon (M1/M2/M3+).
*   Employs **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
*   Designed for **lightweight open-source models** (e.g., `microsoft/Phi-3-mini-4k-instruct`).
*   Includes Python scripts for the complete workflow: data generation -> preparation -> training -> analysis -> quantization -> evaluation -> inference.
    *   `01_generate_qa.py`: Generate initial Q&A pairs.
    *   `02_prepare_data.py`: Format data for training.
    *   `03_train.py`: Run LoRA fine-tuning.
    *   `04_fuse_and_quantize.py`: Merge adapters and quantize.
    *   `05_evaluate.py`: Generate answers on a test/validation set.
    *   `06_inference.py`: Chat interactively with the model.
    *   `visualize_log.py`: Plot training/validation loss curves from logs.
    *   `review_data.py`: Helper script to view data samples.
*   Targets **Dungeons & Dragons 5th Edition (2024 Ruleset)** based on provided source material.
*   Automatic logging of training output to `training_run.log`.

## Project Goals & Future Plans

1.  **Reliable Rules Expert (Current Focus):**
    *   Achieve acceptable accuracy and reliability on D&D 5e rules questions through improved data curation and iterative training/evaluation.
2.  **External Tool Usage:** (Future Goal / Alternative Path)
    *   Rather than train the model to perform complex calculations (e.g., encounter difficulty), integrate existing specialized tools to keep the core LLM focused on rule interpretation and explanation. This could extend to map generation triggers, NPC detail lookups, etc.
3.  **Map Creator/Assistant:** (Future Goal) Develop a model or module focused on generating descriptions or potentially basic layouts for fantasy maps.
4.  **NPC Creator:** (Future Goal) Develop a model or module to assist DMs in generating non-player characters with backstories, personalities, and stats.
5.  **Encounter Builder:** (Future Goal) Develop a model or module to help DMs design balanced and interesting combat or social encounters (potentially via tool integration).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/chanfriendly/dm_assistant
    cd dm_assistant # Or your project directory name
    ```

2.  **Create & Activate Virtual Environment:** (Recommended)
    ```bash
    # Using venv
    python3 -m venv venv
    source venv/bin/activate

    # Or using Conda
    # conda create -n dnd_mlx python=3.12 # Or your preferred Python 3.10+
    # conda activate dnd_mlx
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `mlx`, `mlx-lm>=0.10.0`, `transformers`, `datasets`, `huggingface_hub`, `requests`, `tqdm`, `ollama`, `sentencepiece`, `pyyaml`, `matplotlib`)*

4.  **Prepare Data Sources:**
    *   Obtain Markdown (`.md`) versions of your D&D 5e sources. The Player's Handbook 2024 and Dungeon Master's Guide 2024 are the minimum recommended reference tools, but it is strongly advised to supplement with additional, robust information (Sage Advice Compendium, Monster Manual, etc.).
    *   Place these `.md` files inside the `data/` directory.

5.  **Setup Ollama (Required for Data Generation):**
    *   Install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull a suitable model for generating Q&A pairs (e.g., `llama3.2:latest`).
        ```bash
        ollama pull llama3.2:latest
        ```
    *   Ensure the Ollama server is running before data generation.

## Workflow (Iterative)

**Note:** High-quality results require careful data curation and potentially multiple training iterations. The current model state is unreliable.

1.  **Generate & Curate Q&A Data (`scripts/01_generate_qa.py` & `scripts/review_data.py`):**
    *   Generates raw Q&A pairs from `.md` files using Ollama. Current dataset is ~3700 pairs.
    ```bash
    # Ensure Ollama is running!
    python scripts/01_generate_qa.py --model llama3.2:latest # Or your preferred generator
    ```
    *   **CRITICAL STEP:** Use `scripts/review_data.py` and manual inspection to review and curate the output file (`data/dnd_qa_raw.jsonl`). Remove duplicates, inaccuracies, hallucinations, verbose/low-value answers, and potentially off-topic content. **This requires significant effort and D&D rules knowledge.**
    ```bash
    # Example review command
    python scripts/review_data.py ./data/ -n 50 -r
    ```

2.  **Prepare Data for Training (`scripts/02_prepare_data.py`):**
    *   Formats the cleaned `dnd_qa_raw.jsonl` into `train.jsonl` and `valid.jsonl` in the format expected by `mlx-lm`.
    ```bash
    python scripts/02_prepare_data.py
    ```
    *   Output is saved to `data/processed_dnd_data/`.

3.  **Train the Model (`scripts/03_train.py` & `scripts/visualize_log.py`):**
    *   Performs LoRA fine-tuning on the chosen base model using the processed data. Adjust hyperparameters as needed (batch size, learning rate, iterations, LoRA config).
    ```bash
    # Example training command (adjust iters, batch_size, etc.)
    python scripts/03_train.py --model microsoft/Phi-3-mini-4k-instruct --iters 1000 --batch_size 4 --data ./data/processed_dnd_data --adapter_path ./models/adapters
    ```
    *   Adapters are saved periodically and at the end in `models/adapters/`. Training output is logged to `data/processed_dnd_data/training_run.log`.
    *   **Analyze:** Use `visualize_log.py` to plot the loss curves and identify the best performing checkpoint (lowest validation loss).
    ```bash
    python scripts/visualize_log.py ./data/processed_dnd_data/training_run.log
    ```

4.  **Fuse Adapters and (Optionally) Quantize (`scripts/04_fuse_and_quantize.py`):**
    *   Merges the trained LoRA adapters (from the **best checkpoint identified in Step 3**) with the base model.
    *   Optionally quantizes the fused model. **Evaluate non-quantized first if quantized results are poor.**
    ```bash
    # Example: Fuse only (saves to models/dnd_expert_fused/)
    python scripts/04_fuse_and_quantize.py --model microsoft/Phi-3-mini-4k-instruct --adapter_path ./models/adapters --save_path ./models/dnd_expert_fused --no-quantize --adapter_file <BEST_CHECKPOINT_e.g._0000500_adapters.safetensors>

    # Example: Fuse and Quantize (saves to models/dnd_expert_quantized/)
    python scripts/04_fuse_and_quantize.py --model microsoft/Phi-3-mini-4k-instruct --adapter_path ./models/adapters --save_path ./models/dnd_expert_quantized --adapter_file <BEST_CHECKPOINT_e.g._0000500_adapters.safetensors>
    ```
    *   Specify `--adapter_file` if using a checkpoint other than the final `adapters.safetensors`.

5.  **Evaluate the Model (`scripts/05_evaluate.py`):**
    *   Runs the fine-tuned model (either fused or quantized) against the validation/test set.
    ```bash
    # Evaluate Quantized
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_quantized --test_file ./data/processed_dnd_data/valid.jsonl

    # Evaluate Fused (Non-Quantized)
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_fused --test_file ./data/processed_dnd_data/valid.jsonl
    ```
    *   Outputs results to `evaluation_results/evaluation_outputs.jsonl` for **manual review**. Assess quality based on accuracy, conciseness, and relevance. **If quality is insufficient, return to Step 1 (Data Curation) or Step 3 (Training Hyperparameters).**

6.  **Interact with the Model (`scripts/06_inference.py`):**
    *   Provides a CLI to chat with the *validated* fine-tuned model (use the path to the model version deemed acceptable after evaluation).
    ```bash
    # Example using a validated fused model
    python scripts/06_inference.py --model_path ./models/dnd_expert_fused
    ```

## Customization

*   **Models:** Adjust generator model (Ollama) in `01_generate_qa.py` and base model (Hugging Face ID) in `03_train.py` / `04_fuse_and_quantize.py`.
*   **Data:** **Improving data quality and quantity through generation and rigorous curation is the most impactful customization.** Chunking parameters in `01_generate_qa.py` can also be adjusted.
*   **Training Hyperparameters:** Modify LoRA settings, iterations, batch size, learning rate, sequence length, LR scheduler via YAML config or CLI args in `03_train.py`.
*   **Prompt Template:** Ensure consistency in the prompt format across `02_prepare_data.py`, `05_evaluate.py`, and `06_inference.py`.

## Disclaimer

*   The quality of the fine-tuned model is **highly dependent** on the quality and quantity of the curated Q&A data.
*   **Current evaluation of the quantized model shows it is unreliable and prone to significant hallucinations and errors.** Use with extreme caution and **always verify answers against official sourcebooks.**
*   LLMs can hallucinate, even when fine-tuned. This tool is an assistant, **not** a replacement for rulebooks or DM rulings.
*   Ensure compliance with source material licenses (e.g., OGL for SRD, Fan Content Policy for full books). This project does not distribute copyrighted D&D text.
