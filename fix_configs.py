import yaml
import os
import glob

def update_config(config_path):
    # Load the existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add chat template to model section
    if 'model' in config:
        config['model']['chat_template'] = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'system' %}\n"
            "<|system|>{{ message['content'] }}</system>\n"
            "{% elif message['role'] == 'user' %}\n"
            "<|user|>{{ message['content'] }}</user>\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "<|assistant|>{{ message['content'] }}</assistant>\n"
            "{% endif %}\n"
            "{% endfor %}"
        )
    
    # Add collator to data.train section
    if 'data' in config and 'train' in config['data']:
        config['data']['train']['collator_name'] = "custom_collator.CausalLMDataCollator"
    
    # Write the updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated {config_path}")

# Create the collator file
with open("custom_collator.py", "w") as f:
    f.write("""
from transformers import DefaultDataCollator

class CausalLMDataCollator(DefaultDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        # Set labels equal to input_ids for causal language modeling
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch
""")
    print("Created custom_collator.py")

# Update all config files
for config_file in glob.glob('configs/*.yaml'):
    update_config(config_file)