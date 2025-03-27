# scripts/02_prepare_data.py
import json
import argparse
import os
import random

# --- Configuration ---
RAW_DATA_FILE = "data/dnd_qa_raw.jsonl" # Assuming this is your curated file now
OUTPUT_DIR = "./data/processed_dnd_data" # Directory to save jsonl files
TRAIN_FILENAME = "train.jsonl"
VALID_FILENAME = "valid.jsonl" # Name expected by mlx_lm.lora
TRAIN_TEST_SPLIT = 0.95
# ---

# Prompt format for JSONL structure
def create_jsonl_entry(question: str, answer: str) -> dict:
    """
    Creates a dictionary suitable for JSONL output, using either
    'text' format or 'prompt'/'completion' format.
    'text' format is generally simpler if the model handles the full sequence.
    """
    # Using the 'text' format: concatenating the full prompt + answer
    text_entry = f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.

### Question:
{question}

### Answer:
{answer}</s>""" # EOS token is important for training
    return {"text": text_entry}

    # --- Alternative: 'prompt'/'completion' format ---
    # prompt_part = f"""Below is a question about Dungeons & Dragons 5th Edition rules. Provide a clear and accurate answer based on the official rules.
    #
    # ### Question:
    # {question}
    #
    # ### Answer:"""
    # completion_part = f""" {answer}</s>""" # Add space before answer, include EOS
    # return {"prompt": prompt_part, "completion": completion_part}
    # --- End Alternative ---


def main():
    parser = argparse.ArgumentParser(description="Format raw Q&A data into JSONL for MLX training.")
    parser.add_argument("--raw_file", type=str, default=RAW_DATA_FILE, help="Path to the raw curated JSONL Q&A file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the train.jsonl and valid.jsonl files.")
    parser.add_argument("--split", type=float, default=TRAIN_TEST_SPLIT, help="Train/validation split ratio.")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.raw_file}")

    data = []
    try:
        with open(args.raw_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    qa_pair = json.loads(line)
                    if "question" in qa_pair and "answer" in qa_pair:
                        # Check for non-empty question/answer
                        if qa_pair["question"].strip() and qa_pair["answer"].strip():
                            jsonl_entry = create_jsonl_entry(qa_pair["question"], qa_pair["answer"])
                            data.append(jsonl_entry)
                        else:
                            print(f"Skipping line {line_num+1}: Empty question or answer.")
                    else:
                        print(f"Skipping invalid line {line_num+1}: Missing 'question' or 'answer' key.")
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line {line_num+1}: {line.strip()}")

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
    val_data = data[split_index:]

    print(f"Split: {len(train_data)} training examples, {len(val_data)} validation examples.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training data
    train_path = os.path.join(args.output_dir, TRAIN_FILENAME)
    try:
        with open(train_path, 'w', encoding='utf-8') as f_train:
            for entry in train_data:
                f_train.write(json.dumps(entry) + '\n')
        print(f"Saved training data to: {train_path}")
    except Exception as e:
        print(f"Error writing training file: {e}")

    # Save validation data
    val_path = os.path.join(args.output_dir, VALID_FILENAME)
    try:
        # Only write validation file if there is validation data
        if val_data:
            with open(val_path, 'w', encoding='utf-8') as f_val:
                for entry in val_data:
                    f_val.write(json.dumps(entry) + '\n')
            print(f"Saved validation data to: {val_path}")
        else:
            print("No validation data to save.")
    except Exception as e:
        print(f"Error writing validation file: {e}")

    print("Data preparation complete. Output format: JSONL with 'text' key.")

if __name__ == "__main__":
    main()