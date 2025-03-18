# D&D AI Assistant

A modular AI assistant for Dungeons & Dragons, designed to help Dungeon Masters with rules lookups, NPC creation, map generation, and encounter building.

## Project Overview

This project fine-tunes smaller language models to create specialized D&D assistant modules:

- **Rules Assistant**: Answers D&D rules questions with accurate references
- **NPC Generator**: Creates detailed NPCs with personalities and backstories
- **Map Generator**: Designs battle maps from textual descriptions
- **Encounter Builder**: Creates balanced combat encounters for D&D parties

Each module is trained separately using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters, allowing them to run on consumer hardware (8GB GPU).

## Setup

### Prerequisites

- Python 3.9+
- Git
- A GPU with at least 8GB VRAM (for training)
- D&D sourcebook content in markdown format

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dnd-ai-assistant.git
   cd dnd-ai-assistant
   ```

2. Set up the virtual environment:
   ```bash
   python setup_training.py
   ```

3. Place your D&D markdown files in the `sourcebooks` directory.

## Project Structure

```
.
├── configs/                 # YAML configuration files for training
├── sourcebooks/             # Markdown files of D&D sourcebooks
├── training_data/           # Generated training examples (JSONL format)
├── output/                  # Trained models and checkpoints
├── process_docs.py          # Script to process markdown into training data
├── setup_training.py        # Environment setup script
├── train_all_modules.sh     # Script to run all training jobs
└── README.md                # This file
```

## Usage

### 1. Process Sourcebooks

Process your D&D sourcebooks from markdown into training examples:

```bash
python process_docs.py --input ./sourcebooks --intermediate ./processed_dnd_data.json --output ./training_data
```

### 2. Train Models

Train all modules sequentially:

```bash
bash train_all_modules.sh
```

Or train individual modules:

```bash
source dnd_env/bin/activate
oumi train -c configs/rules_module.yaml
```

### 3. Inference

After training, you can use the models for inference:

```python
from oumi.inference import NativeTextInferenceEngine
from oumi.core.configs import ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

# Load rules module
rules_engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="./output/dnd_rules_module",
        torch_dtype_str="bfloat16"
    )
)

# Create a rules question
conversation = Conversation(messages=[
    Message(role=Role.SYSTEM, content="You are a D&D rules assistant."),
    Message(role=Role.USER, content="How does sneak attack work?")
])

# Get response
result = rules_engine.infer_online([conversation])
print(result[0].messages[-1].content)
```

## Next Steps

Here are the next steps to continue development:

1. **Complete the data processing pipeline**:
   - Add more sourcebooks to the `sourcebooks` directory
   - Run `process_docs.py` to generate training examples
   - Inspect the generated examples to ensure quality

2. **Run training**:
   - Train each module using the configs provided
   - Monitor training progress in `output/<module_name>`
   - Adjust hyperparameters if needed

3. **Evaluate trained models**:
   - Create a test set for each module
   - Evaluate model performance on these test sets
   - Identify areas for improvement

4. **Integration**:
   - Create a unified interface to access all modules
   - Implement a prompt router to direct queries to the appropriate module
   - Build a web or command-line interface

5. **Expansion**:
   - Add more modules (e.g., adventure generator, puzzle creator)
   - Experiment with larger models for improved capabilities
   - Try different PEFT methods like QLoRA

## Troubleshooting

If you encounter configuration errors with Oumi, check the schema:

```bash
oumi config schema --print
```

This will display the full configuration schema with all valid fields and their expected types.

## License

This project is intended for personal use only. The training data derived from D&D books is subject to copyright protection from Wizards of the Coast.