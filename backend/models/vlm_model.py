import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
import os
import pandas as pd
from PIL import Image


class SpaceSiglipModel:

    def __init__(self, model_name="google/siglip2-so400m-patch16-384"):
        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        self.image_embeddings = None
        self.image_paths = None
        self.captions_df = None
        
    def build_or_load_index(self, dataset_dir):
        """Loads dataset CSV and pre-computes or loads image embeddings cache."""
        csv_path = os.path.join(dataset_dir, "vlm_captions_dataset.csv")
        cache_path = os.path.join(os.path.dirname(__file__), "image_embeddings.pt")
        
        if not os.path.exists(csv_path):
            print("Dataset CSV not found. Search functionalities disabled.")
            return

        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        valid_paths = []
        valid_indices = []
        
        # Verify images exist
        for idx, row in df.iterrows():
            img_rel = str(row["image_file_path"]).replace('/', os.sep).replace('\\', os.sep)
            img_full = os.path.join(dataset_dir, img_rel)
            if os.path.exists(img_full):
                valid_paths.append(img_full)
                valid_indices.append(idx)
        
        self.captions_df = df.iloc[valid_indices].reset_index(drop=True)
        self.image_paths = valid_paths
        print(f"Found {len(self.image_paths)} valid images.")
        
        if os.path.exists(cache_path):
            print("Loading cached image embeddings...")
            self.image_embeddings = torch.load(cache_path, map_location=self.device)
            if len(self.image_embeddings) != len(self.image_paths):
                print("Cache size mismatch. Rebuilding index...")
                self.image_embeddings = None
        
        if self.image_embeddings is None:
            print(f"Computing embeddings for {len(self.image_paths)} images...")
            embeddings = []
            batch_size = 16
            for i in range(0, len(self.image_paths), batch_size):
                batch_paths = self.image_paths[i:i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                emb = self.get_embeddings(images=images)
                embeddings.append(emb.cpu())
                print(f"Processed {min(i+batch_size, len(self.image_paths))} / {len(self.image_paths)}")
                
            self.image_embeddings = torch.cat(embeddings, dim=0).to(self.device)
            torch.save(self.image_embeddings, cache_path)
            print(f"Embeddings saved to {cache_path}")

    def search_by_text(self, text, top_k=10):
        if self.image_embeddings is None:
            return []
            
        text_emb = self.get_embeddings(text_list=[text])
        similarities = torch.matmul(text_emb, self.image_embeddings.t()).squeeze(0)
        top_k_vals, top_k_indices = torch.topk(similarities, min(top_k, len(self.image_paths)))
        
        results = []
        for idx in top_k_indices.tolist():
            results.append({
                "path": self.image_paths[idx],
                "title": self.captions_df.iloc[idx]["title"],
                "caption": self.captions_df.iloc[idx]["caption"]
            })
        return results
        
    def search_by_image(self, image, top_k=10):
        if self.image_embeddings is None:
            return "Index not built.", []
            
        img_emb = self.get_embeddings(images=[image])
        similarities = torch.matmul(img_emb, self.image_embeddings.t()).squeeze(0)
        top_k_vals, top_k_indices = torch.topk(similarities, min(top_k, len(self.image_paths)))
        
        results = []
        for idx in top_k_indices.tolist():
            results.append({
                "path": self.image_paths[idx],
                "title": self.captions_df.iloc[idx]["title"],
                "caption": self.captions_df.iloc[idx]["caption"]
            })
            
        best_idx = top_k_indices[0].item()
        best_caption = self.captions_df.iloc[best_idx]["caption"]
        best_title = self.captions_df.iloc[best_idx]["title"]
        caption_text = f"### {best_title}\n\n{best_caption}"
        
        return caption_text, results

    def get_embeddings(self, images=None, text_list=None):
        """
        Takes images (PIL) and/or text (list of strings) and returns SigLIP 2 multi-modal embeddings.
        """
        with torch.no_grad():
            if images is not None and text_list is not None:
                inputs = self.processor(text=text_list, images=images, padding="max_length", return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                image_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                if not isinstance(image_embeds, torch.Tensor):
                    image_embeds = image_embeds.pooler_output
                
                text_kwargs = {"input_ids": inputs["input_ids"]}
                if "attention_mask" in inputs:
                    text_kwargs["attention_mask"] = inputs["attention_mask"]
                text_embeds = self.model.get_text_features(**text_kwargs)
                if not isinstance(text_embeds, torch.Tensor):
                    text_embeds = text_embeds.pooler_output
                
                # Normalize exactly like the paper says
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                return image_embeds, text_embeds
                
            elif images is not None:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                image_embeds = self.model.get_image_features(**inputs)
                if not isinstance(image_embeds, torch.Tensor):
                    image_embeds = image_embeds.pooler_output
                return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                
            elif text_list is not None:
                inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt").to(self.device)
                text_embeds = self.model.get_text_features(**inputs)
                if not isinstance(text_embeds, torch.Tensor):
                    text_embeds = text_embeds.pooler_output
                return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
    def image_text_similarity(self, image, text_candidates):
        """
        Zero-shot capability: Compare an image against a list of text captions to find the best match.
        """
        img_emb, text_embs = self.get_embeddings(images=[image], text_list=text_candidates)
        
        # Calculate logits (probabilities)
        logits_per_image = torch.matmul(img_emb, text_embs.t()) * self.model.logit_scale.exp()
        probs = torch.nn.functional.softmax(logits_per_image, dim=1)
        
        return probs.squeeze().cpu().tolist()


print("SigLip 2 VLM Architecture Ready!")
