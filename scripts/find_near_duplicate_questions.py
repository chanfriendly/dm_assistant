# scripts/find_near_duplicate_questions.py
import json
import argparse
from pathlib import Path
from thefuzz import fuzz # Using thefuzz library
from tqdm import tqdm # For progress bar

def find_near_duplicates(file_path, threshold=90, sample_size=None):
    """
    Finds pairs of questions with high textual similarity using fuzzy matching.

    Args:
        file_path (str or Path): Path to the JSONL file.
        threshold (int): Similarity threshold (0-100). Pairs >= this are flagged.
        sample_size (int, optional): If set, only analyzes a random sample of N questions
                                     to speed up analysis on very large datasets.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return

    questions = []
    print(f"Loading questions from '{file_path.name}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    question = data.get("question")
                    if question:
                        questions.append({"line": i + 1, "text": question})
                    else:
                         print(f"Warning: Skipping line {i+1} due to missing 'question' key.")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if not questions:
        print("No valid questions found.")
        return

    num_questions = len(questions)
    print(f"Loaded {num_questions} questions.")

    # --- Sampling Logic (Optional) ---
    indices_to_check = list(range(num_questions))
    if sample_size and sample_size < num_questions:
        print(f"Sampling {sample_size} questions for analysis...")
        import random
        indices_to_check = random.sample(indices_to_check, sample_size)
        num_questions = sample_size # Adjust for progress bar calculation

    duplicate_pairs = []
    # Calculate total pairs for tqdm progress bar (n * (n-1) / 2)
    total_pairs = num_questions * (num_questions - 1) // 2
    print(f"Comparing pairs (approx. {total_pairs} comparisons)...")

    # Compare each question with every other question *after* it
    with tqdm(total=total_pairs, unit="pair") as pbar:
        for i in range(num_questions):
            idx1 = indices_to_check[i]
            q1_data = questions[idx1]

            for j in range(i + 1, num_questions):
                idx2 = indices_to_check[j]
                q2_data = questions[idx2]

                # Use token_sort_ratio for better handling of word order differences
                similarity = fuzz.token_sort_ratio(q1_data["text"], q2_data["text"])

                if similarity >= threshold:
                    duplicate_pairs.append({
                        "line1": q1_data["line"],
                        "q1": q1_data["text"],
                        "line2": q2_data["line"],
                        "q2": q2_data["text"],
                        "similarity": similarity
                    })
                pbar.update(1) # Update progress bar for each comparison

    print(f"\nFound {len(duplicate_pairs)} pairs with similarity >= {threshold}%:")

    if not duplicate_pairs:
        print("No near-duplicate questions found matching the criteria.")
        return

    # Sort by similarity score (descending) for review priority
    duplicate_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    output_file = file_path.parent / f"near_duplicate_questions_gt_{threshold}_sim.txt"
    print(f"Saving list to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for pair in duplicate_pairs:
            outfile.write(f"--- Similarity: {pair['similarity']:.1f}% ---\n")
            outfile.write(f"L{pair['line1']}: {pair['q1']}\n")
            outfile.write(f"L{pair['line2']}: {pair['q2']}\n\n")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find near-duplicate questions in a JSONL file using fuzzy matching.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL input file (e.g., data/dnd_qa_raw.txt)")
    parser.add_argument("-t", "--threshold", type=int, default=90, help="Similarity threshold (0-100). Pairs at or above this % similarity are flagged.")
    parser.add_argument("-s", "--sample", type=int, default=None, help="Optional: Analyze only a random sample of N questions to speed up processing.")

    args = parser.parse_args()

    if not 0 <= args.threshold <= 100:
        print("Error: Threshold must be between 0 and 100.")
    else:
        find_near_duplicates(args.file_path, args.threshold, args.sample)