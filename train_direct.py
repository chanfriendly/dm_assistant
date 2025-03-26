import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def process_data(dataset, tokenizer, prompt_template):
    """Process the dataset for causal language modeling."""
    def _process(example):
        # Extract the messages
        messages = example["messages"]
        
        # Create a conversation string
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            conversation += f"{role}: {content}\n\n"
        
        # Tokenize
        return tokenizer(conversation, truncation=True, padding="max_length", max_length=512)
    
    # Process all examples
    tokenized_dataset = dataset.map(
        _process,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_module(module_name):
    """Train a specific module."""
    print(f"\n\n{'='*50}")
    print(f"Training {module_name.upper()} module")
    print(f"{'='*50}\n")
    
    # Paths
    data_path = f"./training_data/{module_name}_examples.jsonl"
    output_dir = f"./output/dnd_{module_name}_module"
    os.makedirs(output_dir, exist_ok=True)
    
    # Module-specific settings
    steps = {"rules": 1000, "npc": 500, "map": 300, "encounter": 400}
    lr = {"rules": 2e-5, "npc": 5e-5, "map": 3e-5, "encounter": 3e-5}
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and process dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Dataset size: {len(dataset)} examples")
    
    # Define a basic prompt template
    prompt_template = """
    System: {system}
    User: {user}
    Assistant: {assistant}
    """
    
    # Process the dataset
    processed_dataset = process_data(dataset, tokenizer, prompt_template)
    
    # Data collator for language modeling (handles creating labels from inputs)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr.get(module_name, 3e-5),
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        max_steps=steps.get(module_name, 500),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print(f"Starting training for {steps.get(module_name, 500)} steps...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model(output_dir)
    
    print(f"Training complete for {module_name} module!")

if __name__ == "__main__":
    # Train each module sequentially
    modules = ["rules", "npc", "map", "encounter"]
    
    for module in modules:
        train_module(module)
        print(f"\nFinished training {module} module!\n")