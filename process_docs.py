import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

def extract_text_from_markdown(file_path: str) -> str:
    """Extract text from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_markdown_file(file_path: Path, book_name: str) -> List[Dict[str, Any]]:
    """Process a markdown D&D sourcebook into structured segments."""
    content = extract_text_from_markdown(str(file_path))
    
    # Split content by headings
    # This regex matches markdown headings (# Heading)
    sections = re.split(r'(^|\n)(#{1,6}\s+[^\n]+)', content, flags=re.MULTILINE)
    
    structured_content = []
    current_heading = "Introduction"
    current_content = ""
    
    for section in sections:
        # If this is a heading
        if re.match(r'(^|\n)#{1,6}\s+', section):
            # Save previous section
            if current_content.strip():
                section_data = {
                    "book": book_name,
                    "heading": current_heading,
                    "content": current_content.strip(),
                    "content_type": identify_content_type(current_content)
                }
                structured_content.append(section_data)
            
            # Start new section
            current_heading = section.strip().lstrip('#').strip()
            current_content = ""
        else:
            current_content += section
    
    # Add the final section
    if current_content.strip():
        section_data = {
            "book": book_name,
            "heading": current_heading,
            "content": current_content.strip(),
            "content_type": identify_content_type(current_content)
        }
        structured_content.append(section_data)
    
    return structured_content

def identify_content_type(text: str) -> str:
    """Identify the type of D&D content based on patterns."""
    if re.search(r'(\d+d\d+|Hit: \d+|\| *STR *\| *DEX *\|)', text):
        return "stat_block"
    elif re.search(r'\| *[-:]+ *\|', text):
        return "table"
    elif re.search(r'^\s*\*\*[^*]+\*\*\. ', text, re.MULTILINE):
        return "spell_description"
    elif re.search(r'^\s*\*\*Prerequisite', text, re.MULTILINE):
        return "feat_description"
    elif re.search(r'^\s*\*\*(Armor|Weapon|Tool) Proficiency', text, re.MULTILINE):
        return "class_feature"
    else:
        return "narrative"

def process_all_sourcebooks(directory_path: str, output_path: str) -> List[Dict[str, Any]]:
    """Process all markdown files in a directory into structured content."""
    all_content = []
    
    for file_path in Path(directory_path).glob('*.md'):
        book_name = file_path.stem
        print(f"Processing {book_name}...")
        book_content = process_markdown_file(file_path, book_name)
        all_content.extend(book_content)
    
    # Save structured content
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2)
    
    return all_content

def create_oumi_training_examples(structured_content: List[Dict[str, Any]], 
                                 output_dir: str, 
                                 split_by_module: bool = True) -> None:
    """
    Create training examples in Oumi format from structured content.
    
    Args:
        structured_content: List of structured content items
        output_dir: Directory to save the training examples
        split_by_module: Whether to create separate files for each module
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create examples for each module
    rules_examples = create_rules_examples(structured_content)
    npc_examples = create_npc_examples(structured_content)
    map_examples = create_map_examples(structured_content)
    encounter_examples = create_encounter_examples(structured_content)
    
    # Save examples
    if split_by_module:
        save_examples_to_jsonl(rules_examples, os.path.join(output_dir, "rules_examples.jsonl"))
        save_examples_to_jsonl(npc_examples, os.path.join(output_dir, "npc_examples.jsonl"))
        save_examples_to_jsonl(map_examples, os.path.join(output_dir, "map_examples.jsonl"))
        save_examples_to_jsonl(encounter_examples, os.path.join(output_dir, "encounter_examples.jsonl"))
    else:
        # Combine all examples
        all_examples = rules_examples + npc_examples + map_examples + encounter_examples
        save_examples_to_jsonl(all_examples, os.path.join(output_dir, "all_examples.jsonl"))
    
    print(f"Created {len(rules_examples)} rules examples")
    print(f"Created {len(npc_examples)} NPC examples")
    print(f"Created {len(map_examples)} map examples")
    print(f"Created {len(encounter_examples)} encounter examples")

def save_examples_to_jsonl(examples: List[Dict[str, Any]], file_path: str) -> None:
    """Save examples to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

def create_rules_examples(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create rules examples in Oumi format."""
    examples = []
    
    # Filter for content that likely contains rules
    rule_sections = [item for item in content if any(keyword in item["heading"].lower() 
                    for keyword in ["rule", "combat", "spell", "class", "ability", "skill"])]
    
    for section in rule_sections:
        # Create example queries based on the content
        content_text = section["content"]
        heading = section["heading"]
        
        # Create a question about this rule section
        question = f"How does {heading.lower()} work in D&D?"
        
        # Format as an Oumi conversation example
        example = {
            "messages": [
                {"role": "system", "content": "You are a D&D rules assistant helping a Dungeon Master."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"According to the rules on {heading}:\n\n{content_text[:800]}"}
            ]
        }
        
        examples.append(example)
    
    return examples

def create_npc_examples(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create NPC generation examples in Oumi format."""
    examples = []
    
    # Find sections that might contain character descriptions
    character_sections = [item for item in content if any(keyword in item["heading"].lower() 
                        for keyword in ["npc", "character", "villain", "ally"]) 
                        or "character" in item["content"].lower()]
    
    for section in character_sections:
        # Generate a few different NPC prompts using the section content as inspiration
        races = ["human", "elf", "dwarf", "halfling", "tiefling", "dragonborn", "gnome", "half-orc"]
        roles = ["innkeeper", "guard captain", "noble", "sage", "criminal", "merchant", "priest", "wizard"]
        
        # Take some races and roles from the content if possible
        content_text = section["content"].lower()
        for race in races:
            if race in content_text:
                races = [race] + [r for r in races if r != race][:2]
                break
                
        for role in roles:
            if role in content_text:
                roles = [role] + [r for r in roles if r != role][:2]
                break
        
        # Create a few examples with different prompts
        for race in races[:2]:  # Limit to 2 races
            for role in roles[:2]:  # Limit to 2 roles
                prompt = f"Create a {race} {role} NPC for my D&D campaign."
                
                # Format as an Oumi conversation example
                example = {
                    "messages": [
                        {"role": "system", "content": "You are a D&D NPC creator helping a Dungeon Master."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"Here's a {race} {role} NPC for your campaign:\n\n"
                                                       f"# NPC Description\n\n"
                                                       f"Using elements from {section['book']}, I've created this character. "
                                                       f"This NPC has the following traits and background:\n\n"
                                                       f"- Background inspired by {section['heading']}\n"
                                                       f"- Personality: [Insert personality traits]\n"
                                                       f"- Motivations: [Insert motivations]\n"
                                                       f"- Connections: [Insert connections to the world]\n\n"
                                                       f"# Roleplaying Notes\n\n"
                                                       f"When portraying this character, consider the following:\n\n"
                                                       f"- Voice and mannerisms: [Suggestions]\n"
                                                       f"- Key phrases: [Examples]\n"
                                                       f"- Reactions to players: [Typical responses]"}
                    ]
                }
                
                examples.append(example)
    
    return examples

def create_map_examples(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create map generation examples in Oumi format."""
    examples = []
    
    # Find sections that describe locations that could be mapped
    location_sections = [item for item in content if any(keyword in item["heading"].lower() 
                        or keyword in item["content"].lower()
                        for keyword in ["dungeon", "temple", "cave", "forest", "castle", 
                                       "battlefield", "room", "chamber"])]
    
    for section in location_sections:
        # Extract a description that could be used for map generation
        description = extract_location_description(section["content"])
        if not description:
            continue
        
        # Create a user request for a map
        request = f"Create a battle map for this location: {description}"
        
        # Format as an Oumi conversation example
        example = {
            "messages": [
                {"role": "system", "content": "You are a D&D battle map designer helping a Dungeon Master."},
                {"role": "user", "content": request},
                {"role": "assistant", "content": f"Based on your description, I'll create a battle map for: \"{description}\"\n\n"
                                               f"## Map Description\n\n"
                                               f"This is a 20x20 grid map with the following features:\n\n"
                                               f"- Main features: [List key landscape elements]\n"
                                               f"- Terrain: [Describe terrain types and their grid locations]\n"
                                               f"- Entry/exit points: [Describe where characters can enter/exit]\n"
                                               f"- Cover and obstacles: [List elements providing cover or blocking movement]\n"
                                               f"- Points of interest: [Describe special features]\n\n"
                                               f"## Tactical Considerations\n\n"
                                               f"- Line of sight: [Describe visibility constraints]\n"
                                               f"- Movement challenges: [Note difficult terrain]\n"
                                               f"- Strategic positions: [Highlight advantageous positions]"}
            ]
        }
        
        examples.append(example)
    
    return examples

def create_encounter_examples(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create encounter generation examples in Oumi format."""
    examples = []
    
    # Find sections that might describe encounters or monsters
    monster_sections = [item for item in content if item["content_type"] == "stat_block" or
                        any(keyword in item["heading"].lower() or keyword in item["content"].lower()
                        for keyword in ["monster", "creature", "encounter", "combat", "enemy"])]
    
    for section in monster_sections:
        # Create a request to generate an encounter
        levels = [1, 3, 5, 8, 10, 15]
        party_sizes = [3, 4, 5, 6]
        
        # Create a few examples with different party configurations
        for level in levels[:2]:  # Limit to 2 levels
            for party_size in party_sizes[:2]:  # Limit to 2 party sizes
                request = f"Create a balanced encounter for {party_size} level {level} players."
                
                if "heading" in section and len(section["heading"]) > 0:
                    # If we have a creature name, include it
                    request += f" Include {section['heading']} if appropriate for this level."
                
                # Format as an Oumi conversation example
                example = {
                    "messages": [
                        {"role": "system", "content": "You are a D&D encounter designer helping a Dungeon Master."},
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": f"Here's a balanced encounter for {party_size} level {level} players:\n\n"
                                                       f"## Encounter Summary\n\n"
                                                       f"Difficulty: [Easy/Medium/Hard/Deadly]\n"
                                                       f"Environment: [Appropriate environment]\n"
                                                       f"Theme: [Thematic elements]\n\n"
                                                       f"## Creatures\n\n"
                                                       f"- [Creature 1]: x[quantity] (CR [Challenge Rating])\n"
                                                       f"- [Creature 2]: x[quantity] (CR [Challenge Rating])\n\n"
                                                       f"Total XP: [amount] ([adjusted amount] adjusted for party size)\n\n"
                                                       f"## Tactics\n\n"
                                                       f"[Description of how creatures might behave in combat]\n\n"
                                                       f"## Treasure\n\n"
                                                       f"[Appropriate treasure for this encounter]"}
                    ]
                }
                
                examples.append(example)
    
    return examples

def extract_location_description(content: str) -> Optional[str]:
    """Extract a concise location description suitable for map generation."""
    # Look for paragraphs that contain spatial language
    spatial_terms = ["north", "south", "east", "west", "feet", "ft", "center", "corner", "wall", "door", "entrance"]
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    
    # Score paragraphs based on how suitable they are for map generation
    scored_paragraphs = []
    for p in paragraphs:
        if len(p) < 20 or len(p) > 500:  # Skip very short or long paragraphs
            continue
            
        score = 0
        for term in spatial_terms:
            if term in p.lower():
                score += 1
                
        # Also check for dimensional descriptions like "20 feet"
        if re.search(r'\d+\s*(?:feet|ft|foot|yards|squares)', p.lower()):
            score += 3
            
        if score > 0:
            scored_paragraphs.append((p, score))
    
    # Return the highest scoring paragraph, or None if none found
    if scored_paragraphs:
        return max(scored_paragraphs, key=lambda x: x[1])[0]
    return None

def main():
    """Main function to process D&D sourcebooks and create training examples."""
    parser = argparse.ArgumentParser(description='Process D&D sourcebooks and create training examples.')
    parser.add_argument('--input', '-i', required=True, help='Directory containing markdown sourcebooks')
    parser.add_argument('--intermediate', '-m', required=True, help='Path for intermediate processed JSON')
    parser.add_argument('--output', '-o', required=True, help='Output directory for training examples')
    parser.add_argument('--combined', '-c', action='store_true', help='Create a single combined JSONL file instead of separate files')
    
    args = parser.parse_args()
    
    # Process the sourcebooks
    print(f"Processing markdown files from: {args.input}")
    print(f"Saving processed data to: {args.intermediate}")
    
    content = process_all_sourcebooks(args.input, args.intermediate)
    
    print(f"Processed {len(content)} sections from the sourcebooks.")
    
    # Create training examples
    print(f"Creating training examples in: {args.output}")
    create_oumi_training_examples(content, args.output, not args.combined)
    
    print("Done!")

if __name__ == "__main__":
    main()