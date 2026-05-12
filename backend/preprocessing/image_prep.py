import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class SpaceVLMDataset(Dataset):
    """
    Dataset for the Vision-Language Model. 
    Loads NASA images and captions, and processes them directly for SigLIP 2.
    """
    def __init__(self, csv_file, image_dir, processor, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, os.path.basename(row['image_file_path']))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Fallback to a tiny black image if missing
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        caption = str(row['caption'])
        
        # SigLIP 2 Processor handles BOTH image normalization and text tokenization
        encoding = self.processor(
            images=image, 
            text=caption, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
