# scripts/analyze_topic_balance.py
import json
import argparse
from pathlib import Path
from collections import Counter
import re

# --- Define Keyword Categories ---
# (Expand these lists significantly based on your SRD content and desired granularity)
KEYWORD_CATEGORIES = {
    "Core Mechanics": ["advantage", "disadvantage", "proficiency bonus", "ability check", "saving throw", "attack roll", "action", "bonus action", "reaction", "movement", "speed", "initiative", "hit points", "armor class", "ac", "dc", "condition", "rest", "short rest", "long rest", "cover", "grapple", "shove", "inspiration", "critical hit", "death save", "concentration", "attunement", "multiclass"],
    "Character Classes": ["barbarian", "bard", "cleric", "druid", "fighter", "monk", "paladin", "ranger", "rogue", "sorcerer", "warlock", "wizard", "artificer", "subclass", "archetype", "path", "college", "domain", "circle", "tradition", "oath", "patron", "bloodline"],
    "Races/Species": ["human", "elf", "elves", "dwarf", "dwarves", "halfling", "gnome", "dragonborn", "half-elf", "half-orc", "tiefling", "aarakocra", "loxodon", "grung", "goliath", "orc"], # Add more as needed
    "Conditions": ["blinded", "charmed", "deafened", "exhaustion", "frightened", "grappled", "incapacitated", "invisible", "paralyzed", "petrified", "poisoned", "prone", "restrained", "stunned", "unconscious"],
    "Combat Actions": ["attack", "cast a spell", "dash", "disengage", "dodge", "help", "hide", "ready", "search", "use an object", "grapple", "shove", "opportunity attack", "two-weapon fighting", "unarmed strike"],
    "Magic & Spells": ["spell", "cantrip", "ritual", "spell slot", "spellcasting", "spellbook", "component", "verbal", "somatic", "material", "concentration", "spell level", "school of magic", "abjuration", "conjuration", "divination", "enchantment", "evocation", "illusion", "necromancy", "transmutation"],
    "Magic Items": ["magic item", "potion", "scroll", "wand", "staff", "rod", "ring", "amulet", "armor", "weapon", "shield", "boots", "gloves", "cloak", "helm", "figurine", "ioun stone", "bag of holding", "deck of many"],
    "Equipment & Gear": ["armor", "weapon", "shield", "tool", "artisan's tools", "kit", "mount", "vehicle", "coin", "gold", "gp", "silver", "sp", "copper", "cp", "rations", "rope", "torch"],
    "Monsters": ["monster", "creature type", "aberration", "beast", "celestial", "construct", "dragon", "elemental", "fey", "fiend", "giant", "humanoid", "monstrosity", "ooze", "plant", "undead", "stat block", "challenge rating", "cr", "legendary action", "lair action"],
    "Environment": ["terrain", "difficult terrain", "light", "darkness", "obscured", "cover", "underwater", "falling", "suffocating", "trap", "hazard", "weather", "plane", "planar", "ethereal", "astral"],
    "Social/Misc": ["alignment", "language", "background", "skill", "check", "saving throw", "lifestyle", "downtime", "patron", "npc", "dm", "session zero", "inspiration", "renown", "disease", "poison"],
    # Add more categories and specific keywords
}

# Pre-compile regex patterns for efficiency (optional, simple 'in' check used below)
# KEYWORD_REGEX = {cat: re.compile(r'\b(' + '|'.join(kw) + r')\b', re.IGNORECASE) for cat, kw in KEYWORD_CATEGORIES.items()}

def analyze_balance(file_path):
    """Analyzes keyword frequency to gauge topic balance."""
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return

    category_counts = Counter()
    total_entries = 0

    print(f"\nAnalyzing topics in '{file_path.name}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    question = data.get("question", "").lower()
                    answer = data.get("answer", "").lower()
                    text_content = question + " " + answer
                    total_entries += 1

                    found_categories = set() # Track categories found in this entry

                    for category, keywords in KEYWORD_CATEGORIES.items():
                        for keyword in keywords:
                            # Simple check: keyword boundary might be useful (\b) but requires regex
                            # Using simple 'in' for now, might catch substrings undesirably
                            # Add word boundaries for better matching: f'\\b{keyword}\\b' with regex
                            if f" {keyword} " in f" {text_content} " or text_content.startswith(keyword + " ") or text_content.endswith(" " + keyword):
                                if category not in found_categories:
                                    category_counts[category] += 1
                                    found_categories.add(category)
                                # Optimization: break after finding *any* keyword for this category
                                # if you only want to count entries mentioning the category at all
                                break # Count each entry only once per category

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1}")
                except KeyError:
                     print(f"Warning: Skipping line {i+1} due to missing 'question' or 'answer' key.")

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if total_entries == 0:
        print("No valid entries found to analyze.")
        return

    print("\n--- Topic Distribution Analysis ---")
    print(f"Total Entries Analyzed: {total_entries}\n")
    print("Category Counts (Entries containing keywords from category):")

    # Sort categories by count for better readability
    sorted_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)

    for category, count in sorted_categories:
        percentage = (count / total_entries) * 100
        print(f"- {category:<20}: {count:<5} ({percentage:.1f}%)")

    # Optionally save to file
    output_file = file_path.parent / "topic_balance_analysis.txt"
    print(f"\nSaving analysis results to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("--- Topic Distribution Analysis ---\n")
        outfile.write(f"Total Entries Analyzed: {total_entries}\n\n")
        outfile.write("Category Counts (Entries containing keywords from category):\n")
        for category, count in sorted_categories:
            percentage = (count / total_entries) * 100
            outfile.write(f"- {category:<20}: {count:<5} ({percentage:.1f}%)\n")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze topic balance in a JSONL Q&A file using keywords.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL input file (e.g., data/dnd_qa_raw.txt)")
    args = parser.parse_args()

    analyze_balance(args.file_path)