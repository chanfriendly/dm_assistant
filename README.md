# D&D 5e Rules Expert - MLX Fine-Tuning Project

This project aims to fine-tune a lightweight, open-source large language model (LLM) to become an expert on Dungeons & Dragons 5th Edition rules. The primary focus is on leveraging Apple Silicon hardware using MLX for efficient training and inference.

The goal is to create a helpful assistant for DMs and players that can quickly answer specific rules questions based on the official sourcebooks.

**Current Status (As of 2025-03-27):**

*   Data generation (`01_generate_qa.py`) using Ollama and Markdown source files is functional.
*   Data preparation (`02_prepare_data.py`) into Hugging Face `datasets` format is functional.
*   **Training script (`03_train.py`) is currently under development and troubleshooting.** We are working through compatibility issues with `mlx-lm` version `0.22.2` regarding LoRA parameter application.
*   Scripts for quantization, evaluation, and inference (`04` through `06`) are drafted but untested pending successful training.

## Features

*   Uses **MLX** for hardware-accelerated training and inference on Apple Silicon (M1/M2/M3+).
*   Employs **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
*   Designed for **lightweight open-source models** (e.g., `microsoft/Phi-3-mini-4k-instruct`).
*   Includes Python scripts for:
    *   Generating Question/Answer pairs from Markdown rulebooks using **Ollama**.
    *   Preparing data into Hugging Face `datasets` format.
    *   Training the model via LoRA (Debugging in progress).
    *   Fusing adapters and quantizing the model (Planned).
    *   Evaluating model performance (Planned).
    *   Running interactive inference (Planned).
*   Targets **Dungeons & Dragons 5th Edition (2024 Ruleset)** based on provided source material.

## Project Goals & Future Plans

1.  **Rules Expert (Current Focus):** Complete the fine-tuning and evaluation of a reliable D&D 5e rules expert model.
2.  **Map Creator/Assistant:** Develop a model or module focused on generating descriptions or potentially basic layouts for fantasy maps.
3.  **NPC Creator:** Develop a model or module to assist DMs in generating non-player characters with backstories, personalities, and stats.
4.  **Encounter Builder:** Develop a model or module to help DMs design balanced and interesting combat or social encounters.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/chanfriendly/dm_assistant
    cd dnd_rules_expert
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
    *(Ensure `requirements.txt` includes `mlx`, `mlx-lm>=0.10.0`, `transformers`, `datasets`, `huggingface_hub`, `requests`, `tqdm`, `ollama`, `sentencepiece`)*

4.  **Prepare Data Sources:**
    *   Obtain Markdown (`.md`) versions of the D&D 5e rulebooks you want to train on (e.g., Player's Handbook 2024, Dungeon Master's Guide 2024).
    *   Place these `.md` files inside the `data/` directory. The generation script (`01_generate_qa.py`) looks for `.md` files here.

5.  **Setup Ollama (Required for Data Generation):**
    *   Install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull a suitable model for generating Q&A pairs. `llama3.2` worked well after initial attempts with others failed.
        ```bash
        ollama pull llama3.2:latest
        ```
    *   **Ensure the Ollama server is running** before executing the data generation script (run `ollama serve` in a separate terminal or ensure the Ollama desktop application is active).

## Workflow

1.  **Generate Q&A Data (`scripts/01_generate_qa.py`):**
    *   This script reads the `.md` files in `data/`, chunks them, and uses the specified Ollama model to generate question-answer pairs based on the text.
    ```bash
    # Ensure Ollama is running!
    python scripts/01_generate_qa.py --model llama3.2:latest # Or your preferred generator model
    ```
    *   **CRITICAL STEP:** Manually review and curate the output file (`data/dnd_qa_raw.jsonl`). Remove duplicates, inaccuracies, hallucinations, poorly formatted entries, and low-quality pairs. The quality of the final model *heavily* depends on this cleaning process. Aim for several thousand high-quality pairs.

2.  **Prepare Data for Training (`scripts/02_prepare_data.py`):**
    *   Formats the cleaned `dnd_qa_raw.jsonl` into the instruction format needed by `mlx-lm` and saves it as a Hugging Face `Dataset` object.
    ```bash
    python scripts/02_prepare_data.py
    ```
    *   Output is saved to `data/processed_dnd_data/`.

3.  **Train the Model (`scripts/03_train.py`):**
    *   **[Currently Under Debugging]** This script performs LoRA fine-tuning on the chosen base model using the processed data.
    ```bash
    # Example command (parameters might change based on debugging)
    python scripts/03_train.py --model microsoft/Phi-3-mini-4k-instruct --iters 2000 # Adjust iters
    ```
    *   Upon successful execution, LoRA adapters will be saved in `models/adapters/`.

4.  **Fuse Adapters and Quantize (`scripts/04_fuse_and_quantize.py`):**
    *   Merges the trained LoRA adapters with the base model weights and then quantizes the model (e.g., to 4-bit) for reduced size and potentially faster inference on M1/M2/M3 Macs.
    ```bash
    # Ensure base_model matches the one used in training
    python scripts/04_fuse_and_quantize.py 
    ```
    *   The final quantized model is saved to `models/dnd_expert_quantized/` by default.

5.  **Evaluate the Model (`scripts/05_evaluate.py`):**
    *   Runs the fine-tuned (and potentially quantized) model against the validation set created during data preparation.
    ```bash
    # Evaluate the quantized model
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_quantized 

    # Or evaluate the base model + adapters
    # python scripts/05_evaluate.py --model_path microsoft/Phi-3-mini-4k-instruct --adapter_path ./models/adapters
    ```
    *   Outputs generated answers alongside reference answers to `evaluation_results/evaluation_outputs.jsonl` for manual review and qualitative assessment.

6.  **Interact with the Model (`scripts/06_inference.py`):**
    *   Provides a command-line interface to chat with your trained D&D rules expert.
    ```bash
    python scripts/06_inference.py --model_path ./models/dnd_expert_quantized
    ```
    *   Type your rules questions and press Enter. Use `quit` to exit.

## Customization

*   **Models:**
    *   **Generator Model:** Change `--model` in `01_generate_qa.py` (needs to be an Ollama model). `llama3.2:latest` showed good instruction following.
    *   **Base Model:** Change `--model` in `03_train.py` and `04_fuse_and_quantize.py` (needs to be a Hugging Face identifier compatible with `mlx-lm`, e.g., `microsoft/Phi-3-mini-4k-instruct`).
*   **Data:** The most crucial customization is the **manual review and curation** of `data/dnd_qa_raw.jsonl`. High-quality data is paramount. You can also adjust chunking parameters (`--min_words`, `--max_words`) in `01_generate_qa.py`.
*   **Training Hyperparameters:** Adjust LoRA settings (`--lora_layers`, `--lora_rank`, `--lora_alpha`), training iterations (`--iters`), batch size (`--batch_size`, start with 1 on M1), learning rate (`--learning_rate`), and sequence length (`--max_seq_length`) in `03_train.py`.
*   **Prompt Template:** Ensure the prompt format defined in `02_prepare_data.py`, `05_evaluate.py`, and `06_inference.py` is consistent and suitable for the task.

## Disclaimer

*   The quality of the fine-tuned model is directly dependent on the quality and comprehensiveness of the generated and **manually curated** Q&A data.
*   LLMs can hallucinate or generate incorrect information, even after fine-tuning. **Always** double-check critical rule interpretations against official D&D sourcebooks.
*   This tool is intended as an assistant and quick reference, **not** a replacement for the official rulebooks or the DM's final ruling.
*   Ensure you have the rights to use the D&D source material according to its license (e.g., SRD content under the OGL, or personal use of purchased content). This project does not distribute copyrighted material.