import os
import json
import random
from datasets import load_dataset

def download_wiki_sentences(output_path="data/wiki_sentences.json", count=160):
    print("Downloading Wikitext dataset from Hugging Face...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the standard raw wikitext dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    sentences = []
    for row in dataset:
        text = row['text'].strip()
        # Filter out empty lines, section headers (start with =), and overly massive paragraphs
        if 20 < len(text) < 200 and not text.startswith("="):
            sentences.append(text)
            
    # Sample exactly 160 sentences (seeded for reproducibility)
    random.seed(42)
    selected = random.sample(sentences, count)
    
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
        
    print(f"Successfully saved {count} carrier sentences to {output_path}")

if __name__ == "__main__":
    download_wiki_sentences()