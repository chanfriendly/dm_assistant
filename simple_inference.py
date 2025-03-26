import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU usage to avoid GPU/MPS issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Limit CPU threads

def main():
    print("Loading Phi-2 model (this will take a minute)...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True
    )
    
    # Make sure we have a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with simple configuration
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use full precision
        device_map="cpu"  # Force CPU usage
    )
    
    # Set model to evaluation mode
    model.eval()
    
    print("Model loaded successfully! You can now ask D&D questions.")
    print("Type 'exit' to quit.")
    
    # Create a prompt that simulates the fine-tuning
    dnd_system_prompt = """
You are a helpful D&D assistant with deep knowledge of D&D 5th Edition rules.
Answer questions accurately, citing specific rules when possible.
Keep answers clear and concise while being comprehensive.
"""
    
    while True:
        # Get user input
        user_question = input("\nYour D&D rules question: ")
        
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
            
        # Construct the full prompt
        full_prompt = f"{dnd_system_prompt}\n\nQuestion: {user_question}\n\nAnswer:"
        
        # Tokenize the prompt
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Generate with very conservative settings
        print("Generating response (this may take a moment)...")
        try:
            with torch.no_grad():
                # Use greedy decoding (no sampling) for stability
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 256,  # More explicit length control
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1  # Simple beam search
                )
                
            # Extract the generated text
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            print(f"\nAnswer: {generated_text}")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Please try a different question or rephrase your query.")

if __name__ == "__main__":
    main()