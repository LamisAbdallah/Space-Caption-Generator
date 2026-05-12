import os
import sys
import pandas as pd
import torch
import random
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vlm_model import SpaceSiglipModel

def run_vlm_analysis():
    print("--- Starting VLM Error Analysis & Explainability ---")
    dataset_dir = os.path.join(project_root, "dataset")
    csv_path = os.path.join(dataset_dir, "vlm_captions_dataset.csv")
    
    if not os.path.exists(csv_path):
        print("Dataset not found. Please ensure vlm_captions_dataset.csv exists.")
        return

    # Load Model & Index
    vlm = SpaceSiglipModel()
    vlm.build_or_load_index(dataset_dir)
    
    if vlm.image_embeddings is None:
        print("Failed to load or build VLM index.")
        return

    # 1. Error Analysis: Retrieval@K Benchmark
    print("\n[1] Running Retrieval Error Analysis (Top-K Benchmark)...")
    df = vlm.captions_df
    
    # We will sample 50 random images for a quick benchmark
    num_samples = min(50, len(df))
    sample_indices = random.sample(range(len(df)), num_samples)
    
    top_1_correct = 0
    top_5_correct = 0
    
    errors = []
    
    for idx in sample_indices:
        row = df.iloc[idx]
        actual_image_path = vlm.image_paths[idx]
        query = row['title'] # Using title as search query
        
        # Perform text search
        text_emb = vlm.get_embeddings(text_list=[query])
        similarities = torch.matmul(text_emb, vlm.image_embeddings.t()).squeeze(0)
        _, top_k_indices = torch.topk(similarities, 5)
        
        top_k_paths = [vlm.image_paths[i] for i in top_k_indices.tolist()]
        
        if actual_image_path == top_k_paths[0]:
            top_1_correct += 1
            top_5_correct += 1
        elif actual_image_path in top_k_paths:
            top_5_correct += 1
            # Record an "Error" where it wasn't Top 1
            errors.append((query, actual_image_path, top_k_paths[0]))
        else:
            errors.append((query, actual_image_path, top_k_paths[0]))
            
    print(f"Evaluated {num_samples} samples.")
    print(f"Top-1 Accuracy: {top_1_correct/num_samples:.2%}")
    print(f"Top-5 Accuracy: {top_5_correct/num_samples:.2%}")
    
    if errors:
        print("\nExample Error (Query vs Retrieved Top-1 vs Actual):")
        err = errors[0]
        print(f"Query: '{err[0]}'")
        print(f"Retrieved Result Path: {err[2]}")
        print(f"Actual Ground Truth Path: {err[1]}")

    # 2. Explainability: Semantic Contribution Breakdown
    print("\n[2] VLM Semantic Explainability Check...")
    # Pick one sample to explain
    sample_idx = sample_indices[0]
    sample_title = df.iloc[sample_idx]['title']
    sample_img_path = vlm.image_paths[sample_idx]
    
    print(f"Explaining match for image: {os.path.basename(sample_img_path)}")
    print(f"Full Query: '{sample_title}'")
    
    # Break title into words/sub-phrases to see what the VLM cares about
    words = sample_title.split()
    if len(words) > 2:
        phrases = [" ".join(words[:len(words)//2]), " ".join(words[len(words)//2:])]
        phrases.append(sample_title) # Full phrase
        
        img = Image.open(sample_img_path).convert("RGB")
        try:
            # We use zero-shot image_text_similarity
            probs = vlm.image_text_similarity(img, phrases)
            print("\nContribution (Softmax probability among text segments):")
            for phrase, p in zip(phrases, probs):
                print(f"  - '{phrase}': {p:.1%}")
        except Exception as e:
            print("Could not compute similarity:", e)

if __name__ == "__main__":
    run_vlm_analysis()
