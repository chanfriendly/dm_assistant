from typing import Dict, List, Any
import torch
from transformers import DataCollatorForLanguageModeling

class CausalLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First create a standard batch
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        
        # Set labels equal to input_ids for causal language modeling
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
