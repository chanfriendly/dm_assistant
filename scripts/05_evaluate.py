# scripts/05_evaluate.py
import mlx.core as mx
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import time
import traceback # Import traceback

# Import necessary mlx-lm components
try:
    from mlx_lm.utils import load, generate
    print("Successfully imported load/generate from mlx_lm.utils.")
except ImportError as e:
     print(f"Failed to import mlx_lm components. Check installation. Error: {e}")
     exit()

# Import Hugging Face datasets loader
from datasets import load_from_disk

# Define the prompt template function - MUST MATCH TRAINING
def create_prompt(question: str) -> str:
    """Creates the user prompt for the D&D rules question."""
    return f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.

### Question:
{question}

### Answer:"""

# --- Configuration ---
DEFAULT_MODEL_PATH = Path("./models/dnd_expert_quantized")
DEFAULT_DATA_PATH = Path("./data/processed_dnd_data")
DEFAULT_OUTPUT_FILE = Path("./evaluation_results/evaluation_outputs.jsonl")
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMP = 0.1 # Keep default defined, but won't pass it to generate
# ---

# parse_question_answer function (Keep flexible version)
def parse_question_answer(full_text, index):
    """Extracts question and reference answer from the formatted training text.
       Handles potential indentation and newline variations."""
    try:
        q_marker_indented = "    Question:\n"
        q_marker_unindented = "### Question:\n"
        a_marker_single_nl_indented = "\n    Answer:"
        a_marker_double_nl_indented = "\n\n    Answer:"
        a_marker_single_nl_unindented = "\n### Answer:"
        a_marker_double_nl_unindented = "\n\n### Answer:"
        eos_marker = "</s>"

        q_start = full_text.find(q_marker_indented)
        q_marker = q_marker_indented
        if q_start == -1:
            q_start = full_text.find(q_marker_unindented)
            q_marker = q_marker_unindented
            if q_start == -1: return None, None

        if q_marker == q_marker_indented:
            a_marker_single_nl = a_marker_single_nl_indented
            a_marker_double_nl = a_marker_double_nl_indented
        else:
            a_marker_single_nl = a_marker_single_nl_unindented
            a_marker_double_nl = a_marker_double_nl_unindented

        a_start = full_text.find(a_marker_single_nl, q_start)
        a_marker_len = len(a_marker_single_nl)
        if a_start == -1:
            a_start = full_text.find(a_marker_double_nl, q_start)
            a_marker_len = len(a_marker_double_nl)

        if a_start != -1 and a_start > q_start:
            question = full_text[q_start + len(q_marker) : a_start].strip()
            answer_text_start = a_start + a_marker_len
            if full_text[answer_text_start:].startswith('\n'): answer_text_start += 1
            answer_text = full_text[answer_text_start:].strip()
            eos_pos = answer_text.find(eos_marker)
            answer = answer_text[:eos_pos].strip() if eos_pos != -1 else answer_text
            if not question or not answer: return None, None
            return question, answer
        else:
            return None, None
    except Exception as e:
        print(f"ERROR during parse_question_answer [Example {index}]: {e}")
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned D&D model.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the model directory or HF ID.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapters (Use ONLY if --model_path is BASE model).")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the processed dataset directory.")
    parser.add_argument("--output_file", type=str, default=str(DEFAULT_OUTPUT_FILE), help="File to save evaluation results (JSONL).")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens for answers.")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Temperature for generation (NOTE: Not passed to generate function in this version).") # Updated help text
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluation examples.")

    args = parser.parse_args()

    # Convert paths and handle adapter logic
    model_path_str = args.model_path
    adapter_path_str = args.adapter_path
    data_path = Path(args.data_path)
    output_file = Path(args.output_file)
    if adapter_path_str and Path(model_path_str).is_dir() and Path(model_path_str).name.endswith("quantized"):
         print(f"Warning: Ignoring adapter_path ('{adapter_path_str}') because model_path ('{model_path_str}') appears to be a fused/quantized model.")
         adapter_path_str = None

    print("Starting evaluation...")
    print(f"Model Path: {model_path_str}")
    if adapter_path_str: print(f"Adapter Path: {adapter_path_str}")
    else: print("Adapter Path: None (loading directly from model path)")
    print(f"Data Path: {data_path}")
    print(f"Output File: {output_file}")

    # --- Load Model and Tokenizer ---
    print("\nLoading model and tokenizer...")
    try:
        model, tokenizer = load(
            path_or_hf_repo=model_path_str,
            adapter_path=adapter_path_str
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # --- Load Evaluation Data ---
    print("\nLoading evaluation data...")
    try:
        dataset = load_from_disk(str(data_path))
        # print(f"Dataset loaded. Available splits (keys): {list(dataset.keys())}") # Optional Debug
        split_name = None
        if 'validation' in dataset:
             eval_data = dataset['validation']
             split_name = 'validation'
        elif 'test' in dataset:
             eval_data = dataset['test']
             split_name = 'test'
        else:
              print(f"Error: Neither 'validation' nor 'test' split found in dataset at '{data_path}'. Found keys: {list(dataset.keys())}")
              return
        if args.limit:
            limit = min(args.limit, len(eval_data))
            eval_data = eval_data.select(range(limit))
        print(f"Loaded {len(eval_data)} examples from '{split_name}' split.")

    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        traceback.print_exc()
        return

    # --- Run Generation ---
    print("\nGenerating answers for evaluation set...")
    start_time = time.time()
    results = []
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_count = 0

    try:
        # Clear the file before starting if it exists
        if output_file.exists():
             output_file.unlink()
             print(f"Cleared existing output file: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f_out:
             print(f"Opened {output_file} for writing.")
             for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
                full_text = example.get('text')
                if not full_text:
                     continue

                question, reference_answer = parse_question_answer(full_text, i)

                if question is None or reference_answer is None:
                    continue

                prompt = create_prompt(question)
                generated_answer_raw = "[[GENERATION PENDING]]"
                generated_answer = ""

                try:
                     # **** CORRECTED generate() call - REMOVED temp ****
                     generated_answer_raw = generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        # temp=args.temp, # Removed temp argument
                        verbose=False
                     )

                     if not generated_answer_raw or generated_answer_raw.isspace():
                         generated_answer = "[[EMPTY GENERATION]]"
                     else:
                         generated_answer = generated_answer_raw.strip()
                         # Cleanup logic
                         lower_gen = generated_answer.lower()
                         if lower_gen.startswith("### answer:"): generated_answer = generated_answer[len("### answer:"):]
                         elif lower_gen.startswith("answer:"): generated_answer = generated_answer[len("answer:"):]
                         if "\n### question:" in lower_gen: generated_answer = generated_answer[:lower_gen.find("\n### question:")]
                         if "\nQ:" in generated_answer: generated_answer = generated_answer[:generated_answer.find("\nQ:")]
                         generated_answer = generated_answer.strip()
                         if not generated_answer:
                              generated_answer = "[[EMPTY AFTER CLEANUP]]"

                except Exception as gen_e:
                    print(f"ERROR generating answer for example {i}: {gen_e}")
                    generated_answer = f"[[GENERATION ERROR: {gen_e}]]" # Keep error message

                # Prepare result dictionary
                result = {
                    "id": i,
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": generated_answer,
                }
                results.append(result)

                # Write to file
                try:
                    f_out.write(json.dumps(result) + '\n')
                    write_count += 1
                except Exception as write_e:
                     print(f"ERROR writing result for example {i}: {write_e}")

             print(f"\nDEBUG: Finished processing loop. {write_count} results written.")

    except Exception as e:
         print(f"\nError during generation loop or file writing: {e}")
         traceback.print_exc()

    if write_count == 0 and len(eval_data) > 0:
        print("\nWARNING: No results were successfully written. Check parsing logic or potential generation errors.")

    # --- Print Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nEvaluation complete. {len(results)} results processed, {write_count} written to {output_file}")
    print(f"Total time: {total_time:.2f}s")
    if results:
        avg_time = total_time / len(results) if len(results) > 0 else 0
        print(f"Average time per example processed: {avg_time:.3f}s")

    # --- Next Steps ---
    print("\nNext steps:")
    print(f"1. Manually review the generated answers in '{output_file}'.")
    print("2. Assess accuracy, relevance, and conciseness based on your criteria.")
    print("3. Consider calculating automated metrics (e.g., ROUGE, BLEU) if desired.")

if __name__ == "__main__":
    main()