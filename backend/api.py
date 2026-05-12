from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import os
import io
from PIL import Image

# Import existing models
from models.vlm_model import SpaceSiglipModel
from models.clip_model import SpaceClipModel
from models.caption_model import SpaceCaptionModel
from models.tabular_predictor import NEOHazardPredictor
from models.resnet_transformer_model import SpaceResnetTransformerModel
from models.swin_lstm_model import SpaceSwinLSTMModel

app = FastAPI(title="Space Intelligence Engine API")

# Serve the dataset folder statically so the UI can load the images
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
if os.path.exists(dataset_path):
    app.mount("/dataset", StaticFiles(directory=dataset_path), name="dataset")

# Allow frontend to access the backend directly if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelManager:
    """
    Lazy Loader & Singleton Manager ensuring only ONE heavy model is in VRAM at a time.
    """

    def __init__(self):
        self.active_model_name = None
        self.active_model_instance = None
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")

    def _unload_current(self):
        if self.active_model_instance is not None:
            print(f"Unloading {self.active_model_name} from VRAM...")
            del self.active_model_instance
            self.active_model_instance = None
            self.active_model_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_siglip(self):
        if self.active_model_name != "siglip":
            self._unload_current()
            print("Loading SigLIP into VRAM...")
            self.active_model_instance = SpaceSiglipModel()
            self.active_model_instance.build_or_load_index(self.dataset_dir)
            self.active_model_name = "siglip"
        return self.active_model_instance

    def get_clip(self):
        if self.active_model_name != "clip":
            self._unload_current()
            print("Loading CLIP into VRAM...")
            self.active_model_instance = SpaceClipModel()
            self.active_model_instance.build_or_load_index(self.dataset_dir)
            self.active_model_name = "clip"
        return self.active_model_instance

    def get_captioner(self):
        if self.active_model_name != "caption":
            self._unload_current()
            print("Loading ViT-GPT2 Captioner into VRAM...")
            self.active_model_instance = SpaceCaptionModel()
            self.active_model_name = "caption"
        return self.active_model_instance

    def get_resnet_captioner(self):
        if self.active_model_name != "resnet_caption":
            self._unload_current()
            print("Loading ResNet50+Transformer Captioner into VRAM...")
            self.active_model_instance = SpaceResnetTransformerModel()
            self.active_model_name = "resnet_caption"
        return self.active_model_instance

    def get_swin_lstm_captioner(self):
        if self.active_model_name != "swin_lstm_caption":
            self._unload_current()
            print("Loading Swin-LSTM Captioner into VRAM...")
            self.active_model_instance = SpaceSwinLSTMModel()
            self.active_model_name = "swin_lstm_caption"
        return self.active_model_instance

    def get_neo_predictor(self):
        # Random forest takes zero VRAM, but keeping the pattern intact
        if self.active_model_name != "neo":
            self._unload_current()
            print("Loading NEO Random Forest...")
            self.active_model_instance = NEOHazardPredictor()
            # Try to load existing model
            model_path = os.path.join(os.path.dirname(__file__), "models", "neo_rf_model.pkl")
            if os.path.exists(model_path):
                self.active_model_instance.load_model(model_path)
            self.active_model_name = "neo"
        return self.active_model_instance


# Global Singleton Manager
manager = ModelManager()


@app.post("/api/siglip/search_text")
async def vlm_search_text(
    query: str=Form(...),
    model_id: str=Form("siglip")
):
    try:
        model = manager.get_siglip() if model_id == "siglip" else manager.get_clip()
        raw_results = model.search_by_text(query, top_k=10)
        
        # Convert absolute computer paths to server URLs
        results_urls = []
        for r in raw_results:
            rel_path = os.path.relpath(r["path"], manager.dataset_dir)
            results_urls.append({
                "url": f"http://127.0.0.1:8000/dataset/{rel_path.replace(os.sep, '/')}",
                "title": r["title"],
                "caption": r["caption"]
            })
            
        return {"results": results_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/siglip/search_image")
async def vlm_search_image(
    image: UploadFile=File(...),
    model_id: str=Form("siglip")
):
    try:
        model = manager.get_siglip() if model_id == "siglip" else manager.get_clip()
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        caption, raw_results = model.search_by_image(pil_image, top_k=10)
        
        # Convert absolute computer paths to server URLs
        results_urls = []
        for r in raw_results:
            rel_path = os.path.relpath(r["path"], manager.dataset_dir)
            results_urls.append({
                "url": f"http://127.0.0.1:8000/dataset/{rel_path.replace(os.sep, '/')}",
                "title": r["title"],
                "caption": r["caption"]
            })
            
        return {"caption": caption, "results": results_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caption/generate")
async def generate_caption(
    image: UploadFile=File(...),
    model_id: str=Form("vit_gpt2")
):
    try:
        if model_id == "resnet_transformer":
            model = manager.get_resnet_captioner()
        elif model_id == "swin_lstm":
            model = manager.get_swin_lstm_captioner()
        else:
            model = manager.get_captioner()
            
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Generate entirely new caption
        ai_caption = model.generate_caption(pil_image)
        
        return {"caption": ai_caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/neo/predict")
async def neo_predict(
    h: float=Form(...),
    dia_min: float=Form(...),
    dia_max: float=Form(...),
    vel: float=Form(...),
    miss: float=Form(...)
):
    try:
        model = manager.get_neo_predictor()
        features = {
            'absolute_magnitude_h': h,
            'estimated_diameter_min_km': dia_min,
            'estimated_diameter_max_km': dia_max,
            'relative_velocity_kmh': vel,
            'miss_distance_km': miss
        }
        label, conf, explanation, contributions = model.predict(features)
        return {
            "label": label,
            "confidence": conf,
            "explanation": explanation,
            "contributions": contributions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
