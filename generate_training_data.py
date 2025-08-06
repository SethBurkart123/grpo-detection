#!/usr/bin/env python3

import json
import random
import sys
import os
from pathlib import Path
from rich import print

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from claude.claude_client import claude_chat

def generate_writing_prompts_batch():
    """Generate diverse writing prompts with high entropy."""
    
    # Different themes and approaches to add variety between batches
    themes = [
        "unconventional perspectives and unusual scenarios",
        "experimental formats and creative constraints", 
        "professional and business writing contexts",
        "personal narratives and emotional storytelling",
        "technical and instructional writing",
        "fiction and creative storytelling",
        "journalism and investigative writing",
        "academic and analytical writing",
        "humor and satire",
        "historical and cultural contexts",
        "sci-fi and futuristic scenarios",
        "everyday situations with creative twists",
        "artistic and poetic expressions",
        "social media and modern communication",
        "educational and tutorial content"
    ]
    
    formats = [
        "letters, emails, and correspondence",
        "dialogue and conversations", 
        "reviews and critiques",
        "instructions and how-to guides",
        "stories and narratives",
        "scripts and screenplays",
        "poems and creative writing",
        "essays and articles",
        "lists and structured content",
        "social media posts and tweets",
        "product descriptions and marketing",
        "journal entries and personal writing",
        "interviews and Q&As",
        "proposals and business documents",
        "creative constraints and experimental formats"
    ]
    
    # Randomly select theme and format focus for this batch
    theme_focus = random.choice(themes)
    format_focus = random.choice(formats)
    
    system_prompt = f"""You are an expert at creating diverse, creative writing prompts and exercises. 
Your task is to generate exactly 15 different writing prompts.

For this batch, emphasize: {theme_focus}
And focus particularly on: {format_focus}

But still include variety across:
- Writing styles (creative, technical, persuasive, descriptive, narrative, analytical, etc.)
- Genres (fiction, poetry, screenwriting, journalism, blogging, academic, business, etc.) 
- Purposes (storytelling, instruction, persuasion, entertainment, education, etc.)
- Formats and topics (spanning many different subjects and themes)

Make the prompts highly varied and creative. Avoid repetitive patterns. Each prompt should be unique and interesting.

Output ONLY a JSON array of 15 objects, each with a "prompt" field. No other text.
Example format:
[
  {{"prompt": "Write a product review for a time machine from the perspective of a disappointed customer"}},
  {{"prompt": "Create a dialogue between two characters who can only communicate through metaphors"}},
  ...
]"""

    # Add some randomness to the user message too
    variations = [
        "Generate 15 highly diverse and creative writing prompts. Make them span different genres, styles, formats, and topics. Focus on originality and variety.",
        "Create 15 unique writing exercises that challenge different skills and explore various topics. Emphasize creativity and diversity.",
        "Develop 15 distinct writing prompts that cover a wide range of styles, purposes, and subjects. Make each one interesting and different.",
        "Design 15 varied writing challenges that span multiple genres, formats, and themes. Prioritize uniqueness and creative thinking.",
        "Generate 15 creative writing prompts that explore different approaches, styles, and subject matters. Focus on diversity and originality."
    ]
    
    user_message = random.choice(variations)

    try:
        response = claude_chat(
            message=user_message,
            system_prompt=system_prompt,
            max_tokens=8192,
            model="claude-sonnet-4-20250514"
        )
        
        if not response:
            print("Error: No response from Claude")
            return []
        
        # Try to parse the JSON response
        try:
            prompts_data = json.loads(response.strip())
            if isinstance(prompts_data, list):
                print(json.dumps([item.get("prompt", "") for item in prompts_data if "prompt" in item]))
                return [item.get("prompt", "") for item in prompts_data if "prompt" in item]
            else:
                print("Error: Response is not a JSON array")
                return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response was: {response[:200]}...")
            return []
            
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

def generate_answer_for_prompt(prompt):
    """Just return empty string for answer - we're only generating prompts."""
    return ""

def load_existing_data(filepath):
    """Load existing training data."""
    data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading existing data: {e}")
    return data

def save_data(data, filepath):
    """Save data to JSONL format."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} items to {filepath}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    input_file = "/Users/sethburkart/Documents/Coding/grpo-detection/train_data.jsonl"
    output_file = "/Users/sethburkart/Documents/Coding/grpo-detection/train_data_2.jsonl"
    
    print("Loading existing training data...")
    existing_data = load_existing_data(input_file)
    print(f"Loaded {len(existing_data)} existing examples")
    
    # Keep all existing data (including empty answers)
    print(f"Keeping all existing examples: {len(existing_data)} examples")
    
    # Calculate how many new examples we need
    target_count = 2200
    current_count = len(existing_data)
    needed = max(0, target_count - current_count)
    
    print(f"Target: {target_count} examples")
    print(f"Current: {current_count} examples") 
    print(f"Need to generate: {needed} examples")
    
    if needed == 0:
        print("Already have enough examples!")
        # Just randomize and save
        random.shuffle(existing_data)
        save_data(existing_data, output_file)
        return
    
    # Generate new examples in batches
    new_examples = []
    batches_needed = (needed + 14) // 15  # Round up
    
    print(f"Will generate {batches_needed} batches of 15 prompts each...")
    
    for batch_num in range(batches_needed):
        print(f"\nGenerating batch {batch_num + 1}/{batches_needed}...")
        
        # Generate prompts
        prompts = generate_writing_prompts_batch()
        if not prompts:
            print("Failed to generate prompts for this batch, skipping...")
            continue
            
        print(f"Generated {len(prompts)} prompts")
        
        # Create prompt/answer pairs with empty answers
        for i, prompt in enumerate(prompts):
            if len(new_examples) >= needed:
                break
                
            new_examples.append({
                "prompt": prompt,
                "answer": ""
            })
        
        if len(new_examples) >= needed:
            break
    
    print(f"\nGenerated {len(new_examples)} new examples")
    
    # Combine existing and new data
    all_data = existing_data + new_examples[:needed]
    print(f"Total examples: {len(all_data)}")
    
    # Randomize the order
    print("Randomizing order...")
    random.shuffle(all_data)
    
    # Save the final dataset
    save_data(all_data, output_file)
    print(f"\nFinal dataset saved with {len(all_data)} examples!")

if __name__ == "__main__":
    main()