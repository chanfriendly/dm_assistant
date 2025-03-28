# scripts/review_data.py
import json
import argparse
import random
from pathlib import Path

def review_jsonl(file_path, num_samples=10, show_random=False):
    """Loads and prints samples from a JSONL file."""
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: Data file not found at {file_path}")
        return

    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if not lines:
        print(f"File {file_path} is empty.")
        return

    print(f"\n--- Reviewing {file_path} ({len(lines)} examples total) ---")

    indices_to_show = []
    if show_random:
        if len(lines) <= num_samples:
            indices_to_show = list(range(len(lines)))
        else:
            indices_to_show = random.sample(range(len(lines)), num_samples)
        print(f"Showing {len(indices_to_show)} random samples:")
    else:
        indices_to_show = list(range(min(num_samples, len(lines))))
        print(f"Showing first {len(indices_to_show)} samples:")

    for i in indices_to_show:
        line = lines[i]
        print(f"\nExample {i+1}:")
        try:
            data = json.loads(line)
            # Adjust keys based on your actual data structure (e.g., 'question'/'answer' or 'text')
            if 'text' in data:
                print(data['text'])
            elif 'question' in data and 'answer' in data:
                print(f"Q: {data['question']}")
                print(f"A: {data['answer']}")
            else:
                print(json.dumps(data, indent=2)) # Print full JSON if keys unknown
        except json.JSONDecodeError:
            print(f"  Error decoding JSON: {line.strip()}")
        except Exception as e:
            print(f"  Error processing line: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review samples from JSONL training data.")
    parser.add_argument("data_dir", type=str, help="Directory containing train.jsonl and/or valid.jsonl.")
    parser.add_argument("-n", "--num_samples", type=int, default=10, help="Number of samples to show from each file.")
    parser.add_argument("-r", "--random", action="store_true", help="Show random samples instead of the first N.")
    args = parser.parse_args()

    train_file = Path(args.data_dir) / "train.jsonl"
    valid_file = Path(args.data_dir) / "valid.jsonl"

    if train_file.exists():
        review_jsonl(train_file, args.num_samples, args.random)
    else:
        print(f"Train file not found: {train_file}")

    if valid_file.exists():
        review_jsonl(valid_file, args.num_samples, args.random)
    else:
        print(f"Validation file not found: {valid_file}")