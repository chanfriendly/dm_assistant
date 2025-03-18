import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_path, base_model_name):
    """Load a fine-tuned LoRA model for inference."""
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model."""
    # Format the prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    response = response.split("### Response:")[1].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description='Test a fine-tuned D&D assistant module.')
    parser.add_argument('--model_path', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--base_model', default='mistralai/Mistral-7B-v0.1', help='Base model name')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    print("Model loaded! Enter your questions about D&D rules (or 'exit' to quit):")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        print("\nGenerating response...\n")
        response = generate_response(model, tokenizer, user_input)
        print("Response:", response)

if __name__ == "__main__":
    main()