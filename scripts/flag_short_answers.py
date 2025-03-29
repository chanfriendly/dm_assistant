# scripts/flag_short_answers.py
import json
import argparse
from pathlib import Path

def flag_short_answers(file_path, threshold=10):
    """
    Identifies Q&A pairs with answers shorter than a word threshold.

    Args:
        file_path (str or Path): Path to the JSONL file (e.g., dnd_qa_raw.txt).
        threshold (int): Minimum word count. Answers below this are flagged.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return

    short_answers = []
    print(f"\nScanning '{file_path.name}' for answers with fewer than {threshold} words...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    question = data.get("question", "N/A")
                    answer = data.get("answer", "")

                    word_count = len(answer.split())

                    if word_count < threshold:
                        short_answers.append({
                            "line": i + 1,
                            "word_count": word_count,
                            "question": question,
                            "answer": answer
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1}")
                except KeyError:
                     print(f"Warning: Skipping line {i+1} due to missing 'question' or 'answer' key.")

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    print(f"\nFound {len(short_answers)} answers shorter than {threshold} words:")
    if not short_answers:
        print("No short answers found matching the criteria.")
        return

    # Optionally save to a file or just print
    output_file = file_path.parent / f"short_answers_lt_{threshold}_words.txt"
    print(f"Saving list to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in short_answers:
            outfile.write(f"--- Line: {entry['line']} (Words: {entry['word_count']}) ---\n")
            outfile.write(f"Q: {entry['question']}\n")
            outfile.write(f"A: {entry['answer']}\n\n")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag short answers in a JSONL Q&A file.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL input file (e.g., data/dnd_qa_raw.txt)")
    parser.add_argument("-t", "--threshold", type=int, default=10, help="Word count threshold. Answers strictly less than this will be flagged.")
    args = parser.parse_args()

    flag_short_answers(args.file_path, args.threshold)