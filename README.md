# D&D 5e Rules Expert - MLX Fine-Tuning Project

This project aims to fine-tune a lightweight, open-source large language model (LLM) to become an expert on Dungeons & Dragons 5th Edition rules. The primary focus is on leveraging Apple Silicon hardware using MLX for efficient training and inference.

The goal is to create a helpful assistant for DMs and players that can quickly answer specific rules questions based on the official sourcebooks.

**Current Status (As of 2024-03-29):**

*   Initial project pipeline setup (data generation, preparation, training, quantization, evaluation, inference scripts) is complete.
*   Utility scripts added for training analysis (`visualize_log.py`) and initial data review (`review_data.py`).
*   **Data analysis scripts developed** to aid efficient curation:
    *   `flag_short_answers.py`: Identifies answers below a word count threshold.
    *   `find_near_duplicate_questions.py`: Uses fuzzy matching to find similar questions.
    *   `analyze_topic_balance.py`: Performs keyword analysis to gauge topic distribution.
*   Data generation (`01_generate_qa.py`) using Ollama (`llama3.2`) and Markdown source files is functional.
*   Data preparation (`02_prepare_data.py`) into JSONL format is functional.
*   Training dataset **expanded** to approximately **4,000 Q&A pairs**.
*   Initial **manual data cleaning performed** (e.g., correcting formatting, removing ellipses from answers).
*   Training script (`03_train.py`) updated to automatically log output to a file (`training_run.log` in the data directory).
*   Previous training run completed using LoRA (rank 8, alpha 16) on `microsoft/Phi-3-mini-4k-instruct` with the ~3700 pair dataset and a batch size of 4 for 1000 iterations.
    *   Log visualization confirmed training stability improved with batch size 4, but validation loss plateaued around iteration 500.
*   Previous evaluation of the *quantized* model from the best checkpoint revealed **significant reliability issues (Rated: F)**, including frequent hallucinations and rule inaccuracies.
*   **Next Steps:**
    1.  **Efficient Data Curation:** Utilize the newly developed analysis scripts (`flag_short_answers.py`, `find_near_duplicate_questions.py`, `analyze_topic_balance.py`) to systematically identify and review potential issues (short answers, duplicates, topic gaps) in the ~4,000 pair dataset. Perform targeted manual correction and removal based on script outputs.
    2.  **Evaluate Non-Quantized Model:** Test the *non-quantized* fused model (from the previous best checkpoint, Iter 500) to isolate performance issues potentially caused by quantization vs. the fine-tuning/data itself.
    3.  **Iterate on Training:** Based on evaluation results and data curation, re-train with the improved dataset, potentially adjusting hyperparameters.
    4.  **Root Cause Analysis:** Continue analysis if issues persist after data refinement and testing the non-quantized model.

![loss_curve](https://github.com/user-attachments/assets/6395dd35-9fbb-4db9-a08b-353883f673a7)


## Features

*   Uses **MLX** for hardware-accelerated training and inference on Apple Silicon (M1/M2/M3+).
*   Employs **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
*   Designed for **lightweight open-source models** (e.g., `microsoft/Phi-3-mini-4k-instruct`).
*   Includes Python scripts for the complete workflow: data generation -> preparation -> analysis -> training -> model processing -> evaluation -> inference.
    *   `01_generate_qa.py`: Generate initial Q&A pairs.
    *   `02_prepare_data.py`: Format data for training.
    *   `flag_short_answers.py`: Identifies potentially incomplete answers.
    *   `find_near_duplicate_questions.py`: Flags potentially redundant questions.
    *   `analyze_topic_balance.py`: Checks distribution of topics via keywords.
    *   `03_train.py`: Run LoRA fine-tuning.
    *   `visualize_log.py`: Plot training/validation loss curves from logs.
    *   `04_fuse_and_quantize.py`: Merge adapters and optionally quantize.
    *   `05_evaluate.py`: Generate answers on a test/validation set.
    *   `06_inference.py`: Chat interactively with the model.
    *   `review_data.py`: Helper script to manually view data samples.
*   Targets **Dungeons & Dragons 5th Edition (SRD focused)** based on provided source material.
*   Automatic logging of training output to `training_run.log`.

## Project Goals & Future Plans

1.  **Reliable Rules Expert (Current Focus):**
    *   Achieve acceptable accuracy and reliability on D&D 5e rules questions through improved data curation (using analysis scripts and manual review) and iterative training/evaluation.
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
    *(Ensure `requirements.txt` includes `mlx`, `mlx-lm>=0.10.0`, `transformers`, `datasets`, `huggingface_hub`, `requests`, `tqdm`, `ollama`, `sentencepiece`, `pyyaml`, `matplotlib`, `thefuzz`, `python-Levenshtein`)*

4.  **Prepare Data Sources:**
    *   Obtain Markdown (`.md`) versions of your D&D 5e sources (SRD text provided).
    *   Place these `.md` files inside the `data/` directory.

5.  **Setup Ollama (Required for Initial Data Generation):**
    *   Install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull a suitable model for generating Q&A pairs (e.g., `llama3.2:latest`).
        ```bash
        ollama pull llama3.2:latest
        ```
    *   Ensure the Ollama server is running before data generation.

## Workflow (Iterative)

**Note:** High-quality results require careful data curation and potentially multiple training iterations. The current model state is unreliable.

1.  **Generate Q&A Data (`scripts/01_generate_qa.py`):**
    *   Generates raw Q&A pairs from `.md` files using Ollama. Current dataset is ~4000 pairs.
    ```bash
    # Ensure Ollama is running!
    python scripts/01_generate_qa.py --model llama3.2:latest # Or your preferred generator
    ```
    *   *(Optional but recommended)* Perform initial manual review (`review_data.py`) and cleaning (e.g., removing obvious formatting errors, correcting ellipses).

2.  **Analyze & Refine Data (`scripts/flag_short_answers.py`, `scripts/find_near_duplicate_questions.py`, `scripts/analyze_topic_balance.py`):**
    *   Use analysis scripts to efficiently identify potential issues.
    ```bash
    # Flag answers shorter than 10 words
    python scripts/flag_short_answers.py data/dnd_qa_raw.jsonl -t 10

    # Find questions with >95% similarity (use sampling -s for large files)
    python scripts/find_near_duplicate_questions.py data/dnd_qa_raw.jsonl -t 95

    # Analyze topic keyword distribution (expand keywords in script first!)
    python scripts/analyze_topic_balance.py data/dnd_qa_raw.jsonl
    ```
    *   **CRITICAL STEP:** Manually review the outputs of these scripts. Correct incomplete answers, remove true duplicates/redundancies, address topic imbalances through targeted curation or future generation. **This requires significant effort and D&D rules knowledge.**

3.  **Prepare Data for Training (`scripts/02_prepare_data.py`):**
    *   Formats the *cleaned and refined* data (e.g., `dnd_qa_refined.jsonl`) into `train.jsonl` and `valid.jsonl` expected by `mlx-lm`.
    ```bash
    python scripts/02_prepare_data.py --input_file data/dnd_qa_refined.jsonl # Modify script if needed to take input file arg
    ```
    *   Output is saved to `data/processed_dnd_data/`.

4.  **Train the Model (`scripts/03_train.py` & `scripts/visualize_log.py`):**
    *   Performs LoRA fine-tuning on the chosen base model using the processed, curated data. Adjust hyperparameters as needed.
    ```bash
    # Example training command
    python scripts/03_train.py --model microsoft/Phi-3-mini-4k-instruct --iters <num_iterations> --batch_size 4 --data ./data/processed_dnd_data --adapter_path ./models/adapters
    ```
    *   Training output is logged to `data/processed_dnd_data/training_run.log`.
    *   **Analyze:** Use `visualize_log.py` to plot loss curves and identify the best performing checkpoint (lowest validation loss).
    ```bash
    python scripts/visualize_log.py ./data/processed_dnd_data/training_run.log
    ```

5.  **Fuse Adapters and (Optionally) Quantize (`scripts/04_fuse_and_quantize.py`):**
    *   Merges the trained LoRA adapters (from the **best checkpoint identified in Step 4**) with the base model.
    *   Optionally quantizes. Test non-quantized first if previous quantized results were poor.
    ```bash
    # Fuse non-quantized
    python scripts/04_fuse_and_quantize.py --model <base_model_id> --adapter_file <path_to_best_checkpoint> --save_path ./models/dnd_expert_fused --no-quantize

    # Fuse and Quantize
    python scripts/04_fuse_and_quantize.py --model <base_model_id> --adapter_file <path_to_best_checkpoint> --save_path ./models/dnd_expert_quantized
    ```

6.  **Evaluate the Model (`scripts/05_evaluate.py`):**
    *   Runs the fine-tuned model (fused or quantized) against the validation/test set.
    ```bash
    # Evaluate Fused
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_fused --test_file ./data/processed_dnd_data/valid.jsonl

    # Evaluate Quantized
    python scripts/05_evaluate.py --model_path ./models/dnd_expert_quantized --test_file ./data/processed_dnd_data/valid.jsonl
    ```
    *   Outputs results to `evaluation_results/evaluation_outputs.jsonl` for **manual review**. Assess quality based on accuracy, conciseness, and relevance. **If quality is insufficient, return to Step 2 (Data Refinement) or Step 4 (Training).**

7.  **Interact with the Model (`scripts/06_inference.py`):**
    *   Provides a CLI to chat with the *validated* fine-tuned model (use the path to the model version deemed acceptable after evaluation).
    ```bash
    python scripts/06_inference.py --model_path ./models/dnd_expert_fused # Or dnd_expert_quantized if validated
    ```

## Customization

*   **Models:** Adjust generator model (Ollama) in `01_generate_qa.py` and base model (Hugging Face ID) in `03_train.py` / `04_fuse_and_quantize.py`.
*   **Data:** **Improving data quality and quantity through generation and rigorous curation (aided by analysis scripts) is the most impactful customization.** Chunking parameters in `01_generate_qa.py` can also be adjusted. Keyword lists in `analyze_topic_balance.py` require expansion.
*   **Training Hyperparameters:** Modify LoRA settings, iterations, batch size, learning rate, sequence length, LR scheduler via YAML config or CLI args in `03_train.py`.
*   **Prompt Template:** Ensure consistency in the prompt format across `02_prepare_data.py`, `05_evaluate.py`, and `06_inference.py`.

## Disclaimer

*   The quality of the fine-tuned model is **highly dependent** on the quality and quantity of the curated Q&A data.
*   **Previous evaluation of the quantized model showed it is unreliable and prone to significant hallucinations and errors.** Thorough data refinement and re-evaluation (including non-quantized models) are necessary. Use with extreme caution and **always verify answers against official sourcebooks.**
*   LLMs can hallucinate, even when fine-tuned. This tool is an assistant, **not** a replacement for rulebooks or DM rulings.
*   Ensure compliance with source material licenses (e.g., OGL for SRD, Fan Content Policy for full books). This project does not distribute copyrighted D&D text.
