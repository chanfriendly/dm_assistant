# D&D 5e Rules Expert - MLX Fine-Tuning Project

This project aims to fine-tune a lightweight, open-source large language model (LLM) to become an expert on Dungeons & Dragons 5th Edition rules. The primary focus is on leveraging Apple Silicon hardware using MLX for efficient training and inference.

The goal is to create a helpful assistant for DMs and players that can quickly answer specific rules questions based on the official sourcebooks.

**Current Status (As of 2025-03-27):**

*   Initial project pipeline setup (data generation, preparation, training, quantization, evaluation scripts) is complete.
*   Data generation (`01_generate_qa.py`) using Ollama (`llama3.2`) and Markdown source files is functional.
*   Data preparation (`02_prepare_data.py`) into JSONL format is functional.
*   Training script (`03_train.py`) successfully completed LoRA fine-tuning on a *small initial dataset* (~500 pairs) using `mlx-lm` v0.22.2. Adapters were saved.
*   Fusion and quantization script (`04_fuse_and_quantize.py`) successfully processed the trained adapters and base model into a quantized MLX model.
*   Evaluation script (`05_evaluate.py`) successfully loaded the quantized model and ran inference on the validation set.
*   **Evaluation Results Analysis:** Manual review of the initial evaluation output (`evaluation_results/evaluation_outputs.jsonl`) revealed significant issues with the generated answers, including hallucination, verbosity, and rule inaccuracies. **This indicates the initial small training dataset (~500 pairs) was insufficient for producing a reliable rules expert.**
*   **Next Step:** The immediate priority is to **generate a much larger Q&A dataset** (targeting thousands of pairs) and perform **rigorous manual curation** for accuracy and conciseness before attempting retraining.

## Features

*   Uses **MLX** for hardware-accelerated training and inference on Apple Silicon (M1/M2/M3+).
*   Employs **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
*   Designed for **lightweight open-source models** (e.g., `microsoft/Phi-3-mini-4k-instruct`).
*   Includes Python scripts for the complete workflow: data generation -> preparation -> training -> quantization -> evaluation -> inference.
*   Targets **Dungeons & Dragons 5th Edition (2024 Ruleset)** based on provided source material.

## Project Goals & Future Plans

1.  **Rules Expert (Current Focus):**
    *   Generate and curate a large, high-quality D&D 5e rules Q&A dataset.
    *   Re-train the model using the improved dataset.
    *   Evaluate and iterate until acceptable performance is achieved.
2.  **Map Creator/Assistant:** (Future Goal) Develop a model or module focused on generating descriptions or potentially basic layouts for fantasy maps.
3.  **NPC Creator:** (Future Goal) Develop a model or module to assist DMs in generating non-player characters with backstories, personalities, and stats.
4.  **Encounter Builder:** (Future Goal) Develop a model or module to help DMs design balanced and interesting combat or social encounters.

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
    *(Ensure `requirements.txt` includes `mlx`, `mlx-lm>=0.10.0`, `transformers`, `datasets`, `huggingface_hub`, `requests`, `tqdm`, `ollama`, `sentencepiece`, `pyyaml`)*

4.  **Prepare Data Sources:**
    *   Obtain Markdown (`.md`) versions of the D&D 5e rulebooks (e.g., Player's Handbook 2024, Dungeon Master's Guide 2024).
    *   Place these `.md` files inside the `data/` directory.

5.  **Setup Ollama (Required for Data Generation):**
    *   Install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull a suitable model for generating Q&A pairs (e.g., `llama3.2:latest`).
        ```bash
        ollama pull llama3.2:latest
        ```
    *   Ensure the Ollama server is running before data generation.

## Workflow (Iterative)

**Note:** Steps 1-5 need to be repeated with a significantly larger dataset for improved model quality.

1.  **Generate Q&A Data (`scripts/01_generate_qa.py`):**
    *   Generates raw Q&A pairs from `.md` files using Ollama. **Target thousands of pairs.**
    ```bash
    # Ensure Ollama is running!
    python scripts/01_generate_qa.py --model llama3.2:latest # Or your preferred generator
    ```
    *   **CRITICAL STEP:** Manually review and curate the output file (`data/dnd_qa_raw.jsonl`). Remove duplicates, inaccuracies, hallucinations, verbose/low-value answers. **This requires significant effort.**

2.  **Prepare Data for Training (`scripts/02_prepare_data.py`):**
    *   Formats the cleaned `dnd_qa_raw.jsonl` into `train.jsonl` and `valid.jsonl` in the format expected by `mlx-lm`.
    ```bash
    python scripts/02_prepare_data.py
    ```
    *   Output is saved to `data/processed_dnd_data/`.

3.  **Train the Model (`scripts/03_train.py`):**
    *   Performs LoRA fine-tuning on the chosen base model using the processed data.
    ```bash
    python scripts/03_train.py --model microsoft/Phi-3-mini-4k-instruct --iters <num_iterations> # Adjust iters based on dataset size
    ```
    *   Adapters are saved periodically and at the end in `models/adapters/`.

4.  **Fuse Adapters and Quantize (`scripts/04_fuse_and_quantize.py`):**
    *   Merges the trained LoRA adapters with the base model and quantizes the result.
    ```bash
    python scripts/04_fuse_and_quantize.py --model microsoft/Phi-3-mini-4k-instruct # Ensure matches training
    ```
    *   Final quantized model saved to `models/dnd_expert_quantized/`.

5.  **Evaluate the Model (`scripts/05_evaluate.py`):**
    *   Runs the fine-tuned model against the validation set.
    ```bash
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_quantized
    ```
    *   Outputs results to `evaluation_results/evaluation_outputs.jsonl` for **manual review**. Assess quality based on accuracy, conciseness, and relevance. **Repeat steps 1-5 if quality is insufficient.**

6.  **Interact with the Model (`scripts/06_inference.py`):**
    *   Provides a CLI to chat with the *final validated* D&D rules expert.
    ```bash
    python scripts/06_inference.py --model_path ./models/dnd_expert_quantized
    ```

## Customization

*   **Models:** Adjust generator model (Ollama) in `01_generate_qa.py` and base model (Hugging Face ID) in `03_train.py` / `04_fuse_and_quantize.py`.
*   **Data:** **Improving data quality and quantity through generation and rigorous curation is the most impactful customization.** Chunking parameters in `01_generate_qa.py` can also be adjusted.
*   **Training Hyperparameters:** Modify LoRA settings, iterations, batch size, learning rate, sequence length in `03_train.py`.
*   **Prompt Template:** Ensure consistency in the prompt format across `02_prepare_data.py`, `05_evaluate.py`, and `06_inference.py`.

## Disclaimer

*   The quality of the fine-tuned model is **highly dependent** on the quality and quantity of the curated Q&A data. Initial results with small datasets may be poor.
*   LLMs can hallucinate. **Always verify answers against official sourcebooks.**
*   This tool is an assistant, **not** a replacement for rulebooks or DM rulings.
*   Ensure compliance with source material licenses (e.g., OGL for SRD, Fan Content Policy for full books). This project does not distribute copyrighted D&D text.