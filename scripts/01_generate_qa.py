# scripts/01_generate_qa.py
import json
import os
import glob
import random
import argparse
import re
from tqdm import tqdm
import ollama # Or use openai, anthropic libraries if using their APIs
import time

# --- Configuration ---
DATA_SOURCE_DIR = "./data/" # Directory containing the .md files
OUTPUT_FILE = "./data/dnd_qa_raw.jsonl"
# Chunking strategy: Split by ## or ### headings. Adjust regex if needed.
# Filter chunks that are too short or too long (in words)
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 700 # Tune this based on model context window & desired granularity
NUM_QUESTIONS_PER_CHUNK = 2 # How many questions to try generating per chunk
GENERATION_MODEL = "llama3.2:latest" # Or your preferred Ollama model
# ---

def chunk_markdown_files(dir_path, min_words, max_words):
    """Splits markdown files in a directory into chunks based on headings."""
    chunks = []
    md_files = glob.glob(os.path.join(dir_path, "*.md"))
    print(f"Found {len(md_files)} Markdown files in {dir_path}")

    if not md_files:
         raise FileNotFoundError(f"No .md files found in directory: {dir_path}")

    for filepath in md_files:
        print(f"Processing: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by level 2 or 3 headings, keeping the heading with the following text
            # This regex looks for \n## or \n### followed by a space, and splits there.
            # It uses a positive lookahead (?=\n## |\n### ) to keep the delimiter in the resulting chunks.
            # Simpler split: sections = re.split(r'\n##+ ', content) # Loses heading
            
            # Improved split: Capture heading and split before it
            raw_sections = re.split(r'(\n## |\n### |\n# )', content) # Split and keep delimiters
            
            processed_chunks = []
            current_chunk = ""
            if raw_sections[0]: # Handle content before the first heading
                current_chunk = raw_sections[0].strip()

            for i in range(1, len(raw_sections), 2): # Step by 2 to get delimiter and text
                delimiter = raw_sections[i]
                text_after_delimiter = raw_sections[i+1] if (i+1) < len(raw_sections) else ""
                
                # If current chunk isn't empty, consider it complete
                if current_chunk:
                     processed_chunks.append(current_chunk)
                
                # Start new chunk with the delimiter (heading) and the text after it
                current_chunk = (delimiter + text_after_delimiter).strip()

            if current_chunk: # Add the last chunk
                 processed_chunks.append(current_chunk)

            # Filter chunks by word count
            for chunk in processed_chunks:
                word_count = len(chunk.split())
                if min_words <= word_count <= max_words:
                    chunks.append(chunk)
                # else:
                #     print(f"Skipping chunk with {word_count} words (min:{min_words}, max:{max_words})") # Debug


        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    print(f"Created {len(chunks)} chunks from Markdown files (filtered by word count).")
    return chunks

def generate_qa_pair(client, context, model_name, max_retries=3):
    """Uses an LLM to generate a Q&A pair from context."""
    # Slightly refined prompt
    prompt = f"""
    Given the following text from a Dungeons & Dragons 5th Edition rulebook (likely in Markdown format), please generate exactly one distinct question and a concise, accurate answer based *only* on the provided text. Focus on specific rules, mechanics, or definitions.

    Format your output *exactly* as follows:
    Q: [Your Question Here]
    A: [Your Answer Here]

    Do not include any explanation before the "Q:" or after the "A:". If the text is too fragmented, contains primarily flavor text without clear rules, or doesn't contain a clear rule to ask about, output "Q: N/A\nA: N/A".

    Context:
    ---
    {context}
    ---

    Output:
    Q:""" # Prompt the model to start with Q:

    for attempt in range(max_retries):
        try:
            response = client.generate(
                model=model_name,
                prompt=prompt,
                options={"temperature": 0.2, "stop": ["\n\n", "---", "\nQ:", "\nQuestion:"], "num_predict": 150}, # Stop early, lower temp
                stream=False
            )
            output_text = "Q:" + response['response'].strip()

            if "Q: N/A" in output_text or "A: N/A" in output_text:
                # print(f"Skipping chunk due to N/A response.") # Debug N/A chunks
                return None

            q_start = output_text.find("Q:")
            a_start = output_text.find("\nA:") # Look for newline before A:

            if q_start != -1 and a_start != -1 and a_start > q_start:
                question = output_text[q_start + 2:a_start].strip()
                answer = output_text[a_start + 3:].strip() # Skip "\nA:"

                # Basic validation
                if question and answer and len(question) > 10 and len(answer) > 5 and '?' in question:
                    # print(f"Generated Q: {question}") # Debug
                    return {"question": question, "answer": answer}
                else:
                     if attempt == max_retries - 1: # Only print warning on last attempt
                        print(f"Warning: Invalid Q/A format/content (Attempt {attempt+1}): {output_text}")

            else:
                if attempt == max_retries - 1: # Only print warning on last attempt
                    print(f"Warning: Could not parse Q/A structure (Attempt {attempt+1}): {output_text}")

        except Exception as e:
            print(f"Error during generation (attempt {attempt+1}/{max_retries}): {e}")
            # Add a small delay before retrying, especially for API calls or overloaded local server
            time.sleep(0.5 * (attempt + 1)) 
    
    print(f"Failed to generate valid Q/A after {max_retries} retries for context chunk.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from D&D Markdown files using an LLM.")
    parser.add_argument("--data_dir", type=str, default=DATA_SOURCE_DIR, help="Directory containing the source .md files.")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE, help="Path to save the generated JSONL file.")
    parser.add_argument("--min_words", type=int, default=MIN_CHUNK_WORDS, help="Min word count for text chunks.")
    parser.add_argument("--max_words", type=int, default=MAX_CHUNK_WORDS, help="Max word count for text chunks.")
    parser.add_argument("--num_questions", type=int, default=NUM_QUESTIONS_PER_CHUNK, help="Number of Q/A pairs to generate per chunk.")
    parser.add_argument("--model", type=str, default=GENERATION_MODEL, help="Ollama model name (e.g., llama3.2:latest).")
    args = parser.parse_args()

    print("Starting Q&A generation...")
    print(f"Using model: {args.model}")
    print(f"Markdown source directory: {args.data_dir}")
    print(f"Outputting to: {args.output_file}")

    # Initialize the client (Ollama example)
    try:
        client = ollama.Client()
        client.list()
        print(f"Successfully connected to Ollama and model '{args.model}' seems available.")
    except Exception as e:
        print(f"Error connecting to Ollama or finding model '{args.model}'. Is Ollama running? Is the model pulled? Error: {e}")
        return

    chunks = chunk_markdown_files(args.data_dir, args.min_words, args.max_words)
    if not chunks:
         print("No suitable chunks found. Check markdown parsing or word count filters.")
         return
         
    qa_pairs = []
    processed_chunks_count = 0

    print(f"\nAttempting to generate Q&A pairs from {len(chunks)} chunks...")
    # Use tqdm for progress bar
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        generated_count = 0
        attempts = 0
        # Allow more attempts per chunk since generation can fail
        max_attempts_per_chunk = args.num_questions * 3 
        
        chunk_questions = [] # Store questions generated from this chunk to avoid duplicates
        
        while generated_count < args.num_questions and attempts < max_attempts_per_chunk:
            attempts += 1
            qa_pair = generate_qa_pair(client, chunk, args.model)
            if qa_pair:
                # Basic check for duplicates within the chunk and globally (can be slow for large datasets)
                # A better approach might use semantic similarity if needed, but simple check is okay for now
                is_duplicate = False
                if qa_pair["question"] in chunk_questions:
                    is_duplicate = True
                else:
                    # Optional: Check against all previous pairs (can become slow)
                    # for existing_pair in qa_pairs:
                    #     if existing_pair["question"] == qa_pair["question"]:
                    #         is_duplicate = True
                    #         break
                    pass 

                if not is_duplicate:
                    qa_pairs.append(qa_pair)
                    chunk_questions.append(qa_pair["question"])
                    generated_count += 1
            # Optional: Add a small delay if hitting rate limits or high load
            # time.sleep(0.1) 
        processed_chunks_count +=1

    print(f"\nProcessed {processed_chunks_count} chunks.")
    print(f"Generated {len(qa_pairs)} Q&A pairs.")

    # Save to JSONL
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"Saved Q&A pairs to {args.output_file}")
    except Exception as e:
         print(f"Error saving output file: {e}")


    print("\n--- IMPORTANT ---")
    print(f"Please MANUALLY REVIEW the generated data in '{args.output_file}'.")
    print("Quality depends heavily on the generator model and chunking. Clean bad/irrelevant/duplicate entries.")
    print("Aim for several thousand high-quality pairs if possible.")


if __name__ == "__main__":
    main()