# scripts/05_evaluate.py
import mlx.core as mx
# Ensure correct imports
try:
    from mlx_lm.utils import load, generate
except ImportError as e:
     print(f"Failed to import mlx-lm components. Check installation. Error: {e}")
     exit()
     
import argparse
import os
import json
from datasets import load_from_disk
from tqdm import tqdm
import time

# Define the prompt template function - MUST MATCH TRAINING/INFERENCE
def create_prompt(question: str) -> str:
    return f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.

### Question:
{question}

### Answer:"""

# --- Configuration ---
DEFAULT_MODEL_PATH = "../models/dnd_expert_quantized" # Path to the quantized model
DEFAULT_DATA_PATH = "../data/processed_dnd_data"
DEFAULT_OUTPUT_FILE = "../evaluation_results/evaluation_outputs.jsonl"
DEFAULT_MAX_TOKENS = 300 # Max tokens for generated answers
# ---

def parse_question_answer(full_text):
    """Extracts question and reference answer from the formatted text."""
    try:
        q_marker = "### Question:\n"
        a_marker = "\n\n### Answer:\n"
        eos_marker = "</s>" # Check for EOS token if used in prepare_data

        q_start = full_text.find(q_marker)
        a_start = full_text.find(a_marker)
        
        if q_start != -1 and a_start != -1 and a_start > q_start:
             question = full_text[q_start + len(q_marker):a_start].strip()
             # Find end of answer (either end of string or EOS token)
             answer_text = full_text[a_start + len(a_marker):].strip()
             eos_pos = answer_text.find(eos_marker)
             if eos_pos != -1:
                 answer = answer_text[:eos_pos].strip()
             else:
                 answer = answer_text # Assume answer goes to the end if no EOS
                 
             return question, answer
        else:
             print(f"Warning: Could not parse Q/A markers in: {full_text[:150]}...")
             return None, None
    except Exception as e:
        print(f"Error parsing text: {e} - Text: {full_text[:150]}...")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned D&D model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the fine-tuned (and possibly quantized) model directory OR base model ID if using adapters.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapters (use if model_path is base model).")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to the processed dataset directory (containing 'validation' split).")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="File to save evaluation results (JSONL).")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens to generate for answers.")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for generation.")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True, help="Trust remote code (e.g., for Phi-3).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluation examples.")

    args = parser.parse_args()

    print("Starting evaluation...")
    print(f"Model: {args.model_path}")
    if args.adapter_path:
         print(f"Adapters: {args.adapter_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_file}")

    # --- Load Model and Tokenizer ---
    print("\nLoading model and tokenizer...")
    try:
        model, tokenizer = load(
            args.model_path,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Load Evaluation Data ---
    print("\nLoading evaluation data...")
    try:
        dataset = load_from_disk(args.data_path)
        # Ensure 'validation' split exists, or use 'test' if named differently
        if 'validation' not in dataset:
             if 'test' in dataset:
                 eval_data = dataset['test']
                 print("Using 'test' split for evaluation.")
             else:
                  print("Error: Neither 'validation' nor 'test' split found in dataset.")
                  return
        else:
             eval_data = dataset['validation']
             
        if args.limit:
            eval_data = eval_data.select(range(min(args.limit, len(eval_data)))) # Ensure limit doesn't exceed dataset size
        print(f"Loaded {len(eval_data)} evaluation examples.")
    except Exception as e:
        print(f"Error loading dataset from {args.data_path}: {e}")
        return
        
    # --- Run Generation ---
    results = []
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print("\nGenerating answers for evaluation set...")
    start_time = time.time()
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
             for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
                full_text = example.get('text') # Use .get for safety
                if not full_text:
                     print(f"Skipping example {i} due to missing 'text' field.")
                     continue

                question, reference_answer = parse_question_answer(full_text)

                if question is None or reference_answer is None:
                    print(f"Skipping example {i} due to parsing error.")
                    continue

                # Apply the *same* prompt template used during training
                prompt = create_prompt(question)
                
                generated_answer = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temp=args.temp,
                    verbose=False
                )

                # Clean up potential model self-correction/repetition
                generated_answer = generated_answer.strip()
                # More robust cleanup: Check if answer starts repeating the question/prompt
                lower_gen = generated_answer.lower()
                if lower_gen.startswith("### answer:"):
                     generated_answer = generated_answer[len("### answer:"):].strip()
                if lower_gen.startswith("answer:"):
                     generated_answer = generated_answer[len("answer:"):].strip()
                # Stop if it starts asking another question
                if "\n### question:" in lower_gen:
                     generated_answer = generated_answer[:lower_gen.find("\n### question:")].strip()


                result = {
                    "id": i,
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": generated_answer,
                }
                results.append(result)
                f_out.write(json.dumps(result) + '\n')
    except Exception as e:
         print(f"\nError during generation or writing output: {e}")
         import traceback
         traceback.print_exc()


    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(results) if results else 0

    print(f"\nEvaluation complete. {len(results)} results saved to {args.output_file}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per example: {avg_time:.3f}s")

    # --- Next Steps ---
    print("\nNext steps:")
    print(f"1. Manually review the generated answers in '{args.output_file}'.")
    print("2. Calculate metrics if desired (manual accuracy, ROUGE, BLEU).")
    
if __name__ == "__main__":
    main()