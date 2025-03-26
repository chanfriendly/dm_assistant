import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_and_use_model():
    # Get the current directory where the script is run from
    current_dir = os.getcwd()
    
    # Check if we're running from inside the model directory or from project root
    if os.path.basename(current_dir) == "dnd_rules_module":
        adapter_path = "."  # We're already in the model directory
    else:
        adapter_path = "./output/dnd_rules_module"  # Path from project root
    
    # Check if adapter_config.json exists in the specified path
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Error: adapter_config.json not found in {adapter_path}")
        print("Current directory:", current_dir)
        print("Files in this directory:", os.listdir("."))
        return
    
    # Base model
    base_model = "microsoft/phi-2"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return
    
    # Load adapter
    print(f"Loading adapter from {adapter_path}")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print(f"Looking for adapter_config.json in: {os.path.abspath(adapter_path)}")
        return
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded! You can now ask D&D rules questions.")
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("\nAsk a D&D rules question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Format the input as a system+user prompt
        prompt = f"System: You are a D&D rules assistant helping a Dungeon Master.\nUser: {user_input}\nAssistant:"
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate a response
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    load_and_use_model()