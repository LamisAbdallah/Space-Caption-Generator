import torch
import os
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util


class SpaceClipModel:

    def __init__(self, model_name='clip-ViT-B-32'):
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.image_embeddings = None
        self.image_paths = None
        self.captions_df = None
        
    def build_or_load_index(self, dataset_dir):
        """Loads dataset CSV and pre-computes or loads image embeddings cache."""
        csv_path = os.path.join(dataset_dir, "vlm_captions_dataset.csv")
        cache_path = os.path.join(os.path.dirname(__file__), "clip_image_embeddings.pt")
        
        if not os.path.exists(csv_path):
            print("Dataset CSV not found. Search functionalities disabled.")
            return

        print("Loading dataset for CLIP...")
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
            print("Loading cached CLIP image embeddings...")
            self.image_embeddings = torch.load(cache_path, map_location=self.device)
            if len(self.image_embeddings) != len(self.image_paths):
                print("Cache size mismatch. Rebuilding index...")
                self.image_embeddings = None
        
        if self.image_embeddings is None:
            print(f"Computing embeddings for {len(self.image_paths)} images...")
            self.image_embeddings = self.model.encode(
                [Image.open(p).convert("RGB") for p in self.image_paths],
                batch_size=16,
                convert_to_tensor=True,
                show_progress_bar=True
            ).to(self.device)
            
            torch.save(self.image_embeddings, cache_path)
            print(f"Embeddings saved to {cache_path}")

    def search_by_text(self, text, top_k=10):
        if self.image_embeddings is None:
            return []
            
        query_emb = self.model.encode([text], convert_to_tensor=True).to(self.device)
        hits = util.semantic_search(query_emb, self.image_embeddings, top_k=top_k)[0]
        
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            results.append({
                "path": self.image_paths[idx],
                "title": self.captions_df.iloc[idx]["title"],
                "caption": self.captions_df.iloc[idx]["caption"]
            })
        return results
        
    def search_by_image(self, image, top_k=10):
        if self.image_embeddings is None:
            return "Index not built.", []
            
        query_emb = self.model.encode([image], convert_to_tensor=True).to(self.device)
        hits = util.semantic_search(query_emb, self.image_embeddings, top_k=top_k)[0]
        
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            results.append({
                "path": self.image_paths[idx],
                "title": self.captions_df.iloc[idx]["title"],
                "caption": self.captions_df.iloc[idx]["caption"]
            })
            
        best_idx = hits[0]['corpus_id']
        best_caption = self.captions_df.iloc[best_idx]["caption"]
        best_title = self.captions_df.iloc[best_idx]["title"]
        caption_text = f"### {best_title}\n\n{best_caption}"
        
        return caption_text, results


print("CLIP (Sentence-Transformers) Ready!")
