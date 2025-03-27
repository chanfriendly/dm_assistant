# scripts/02_prepare_data.py
import json
import argparse
import os
from datasets import Dataset, DatasetDict
import random

# --- Configuration ---
RAW_DATA_FILE = "./data/dnd_qa_raw.jsonl"
OUTPUT_DIR = "./data/processed_dnd_data"
TRAIN_TEST_SPLIT = 0.95 # 95% for training, 5% for evaluation
# ---

# Define the prompt template function
# This should match the template used during inference!
def create_full_prompt_with_answer(question: str, answer: str) -> str:
     # This format is often used by mlx-lm's training script directly
     # It concatenates instruction, input (if any), and output (answer)
    return f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.

    Question:
    {question}

    Answer:
    {answer}"""

def create_full_prompt_with_answer(question: str, answer: str) -> str:
     # This format is often used by mlx-lm's training script directly
     # It concatenates instruction, input (if any), and output (answer)
    return f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.

    Question:
    {question}

    Answer:
    {answer}</s>""" 
    
def main():
    parser = argparse.ArgumentParser(description="Format raw Q&A data for MLX training.")
    parser.add_argument("--raw_file", type=str, default=RAW_DATA_FILE, help="Path to the raw JSONL Q&A file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the processed data (Hugging Face Dataset format).")
    parser.add_argument("--split", type=float, default=TRAIN_TEST_SPLIT, help="Train/test split ratio.")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.raw_file}")
    
    data = []
    try:
        with open(args.raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    qa_pair = json.loads(line)
                    if "question" in qa_pair and "answer" in qa_pair:
                        # Format for mlx-lm: it expects a single "text" field 
                        # containing the full prompt + answer for Causal LM training.
                        full_text = create_full_prompt_with_answer(qa_pair["question"], qa_pair["answer"])
                        data.append({"text": full_text}) 
                        # Alternatively, keep separate fields if your training script handles it:
                        # data.append({
                        #     "instruction": "Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.",
                        #     "input": qa_pair["question"], # Treat question as input
                        #     "output": qa_pair["answer"]
                        # })
                    else:
                        print(f"Skipping invalid line: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}")
        
        print(f"Loaded {len(data)} valid Q&A pairs.")

    except FileNotFoundError:
        print(f"Error: Raw data file not found at {args.raw_file}")
        return

    if not data:
        print("Error: No valid data loaded. Exiting.")
        return

    # Shuffle the data
    random.shuffle(data)

    # Split data
    split_index = int(len(data) * args.split)
    train_data = data[:split_index]
    test_data = data[split_index:]

    print(f"Split: {len(train_data)} training examples, {len(test_data)} evaluation examples.")

    # Create Hugging Face Datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': test_dataset # Use 'validation' or 'test' as needed by mlx-lm train script
    })

    # Save the dataset
    print(f"Saving processed dataset to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
