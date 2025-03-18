#!/bin/bash
# Create directories
mkdir -p configs
mkdir -p output
# Save YAML files to configs directory
cat > configs/rules_module.yaml << 'EOL'
model:
  model_name: "microsoft/phi-2"
  trust_remote_code: true
  tokenizer_kwargs:
    pad_token: "<pad>"
  torch_dtype_str: "bfloat16"
  model_kwargs:
    load_in_4bit: true
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "./training_data/rules_examples.jsonl"
training:
  output_dir: "output/dnd_rules_module"
  optimizer: "adamw_torch_fused"
  learning_rate: 2e-5
  max_steps: 1000
  use_peft: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 10
  save_steps: 100
  eval_strategy: "no"
  trainer_kwargs:
   label_names: [] 
peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
EOL
cat > configs/npc_module.yaml << 'EOL'
model:
  model_name: "microsoft/phi-2"
  trust_remote_code: true
  tokenizer_kwargs:
    pad_token: "<pad>"
  torch_dtype_str: "bfloat16"
  model_kwargs:
    load_in_4bit: true
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "./training_data/npc_examples.jsonl"
training:
  output_dir: "output/dnd_npc_module"
  optimizer: "adamw_torch_fused"
  learning_rate: 5e-5
  max_steps: 500
  use_peft: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 10
  save_steps: 100
  eval_strategy: "no"
  trainer_kwargs:
   label_names: [] 
peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
EOL
cat > configs/map_module.yaml << 'EOL'
model:
  model_name: "microsoft/phi-2"
  trust_remote_code: true
  tokenizer_kwargs:
    pad_token: "<pad>"
  torch_dtype_str: "bfloat16"
  model_kwargs:
    load_in_4bit: true
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "./training_data/map_examples.jsonl"
training:
  output_dir: "output/dnd_map_module"
  optimizer: "adamw_torch_fused"
  learning_rate: 3e-5
  max_steps: 300
  use_peft: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 10
  save_steps: 50
  eval_strategy: "no"
  trainer_kwargs:
   label_names: [] 
peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
EOL
cat > configs/encounter_module.yaml << 'EOL'
model:
  model_name: "microsoft/phi-2"
  trust_remote_code: true
  tokenizer_kwargs:
    pad_token: "<pad>"
  torch_dtype_str: "bfloat16"
  model_kwargs:
    load_in_4bit: true
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "./training_data/encounter_examples.jsonl"
training:
  output_dir: "output/dnd_encounter_module"
  optimizer: "adamw_torch_fused"
  learning_rate: 3e-5
  max_steps: 400
  use_peft: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  logging_steps: 10
  save_steps: 50
  eval_strategy: "no"
  trainer_kwargs:
   label_names: [] 
peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
EOL


# Activate the virtual environment
source dnd_env/bin/activate

# Process the sourcebooks and create training data
echo "Processing sourcebooks and creating training data..."
python process_docs.py --input ./sourcebooks --intermediate ./processed_dnd_data.json --output ./training_data

# Train each module
echo "Training rules module..."
oumi train -c configs/rules_module.yaml

echo "Training NPC module..."
oumi train -c configs/npc_module.yaml

echo "Training map module..."
oumi train -c configs/map_module.yaml

echo "Training encounter module..."
oumi train -c configs/encounter_module.yaml

echo "All training complete!"