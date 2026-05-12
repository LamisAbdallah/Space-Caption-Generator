import gradio as gr
import os
import sys
import torch
from PIL import Image

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vlm_model import SpaceSiglipModel
from models.tabular_predictor import NEOHazardPredictor

# Initialize models (if available)
print("Loading models... (This might take a minute)")
try:
    vlm_model = SpaceSiglipModel()
    # Build or load the image index for search
    dataset_dir = os.path.join(project_root, "dataset")
    vlm_model.build_or_load_index(dataset_dir)
except Exception as e:
    print(f"Warning: Failed to load VLM model - {e}")
    vlm_model = None

try:
    neo_predictor = NEOHazardPredictor()
    neo_model_path = os.path.join(project_root, "models", "neo_rf_model.pkl")
    neo_csv_path = os.path.join(project_root, "dataset", "neo_hazard_dataset.csv")
    
    if os.path.exists(neo_model_path):
        neo_predictor.load_model(neo_model_path)
    elif os.path.exists(neo_csv_path):
        # Auto-train if dataset exists but model doesn't
        print("No trained model found. Auto-training from dataset...")
        X_train, X_test, y_train, y_test = neo_predictor.prepare_data(neo_csv_path)
        neo_predictor.train(X_train, y_train)
        metrics = neo_predictor.evaluate(X_test, y_test)
        neo_predictor.save_model(neo_model_path)
    else:
        print("NEO dataset not found. Run: python scraping/scraping_neo.py")
except Exception as e:
    print(f"Warning: Failed to load NEO Predictor - {e}")
    neo_predictor = None


# --- VLM Functions ---
def text_to_image_search(search_query):
    if vlm_model is None:
        return []
    if not search_query.strip():
        return []
    
    images = vlm_model.search_by_text(search_query, top_k=10)
    return images

def image_to_image_search(image):
    if vlm_model is None:
        return "Model not loaded properly.", []
    if image is None:
        return "Please upload an image.", []
    
    caption, images = vlm_model.search_by_image(image, top_k=10)
    return caption, images

# --- NEO Hazard Prediction Functions ---
def predict_hazard(abs_mag, diam_min, diam_max, velocity, miss_dist):
    if neo_predictor is None or not neo_predictor.is_trained:
        return "Model not loaded or not trained.", "Please train the model first. Run: `python training/train_baseline.py`", None
    
    features = {
        'absolute_magnitude_h': abs_mag,
        'estimated_diameter_min_km': diam_min,
        'estimated_diameter_max_km': diam_max,
        'relative_velocity_kmh': velocity,
        'miss_distance_km': miss_dist,
    }
    
    try:
        label, confidence, explanation, contributions = neo_predictor.predict(features)
        result_text = f"{label}\nConfidence: {confidence:.1%}"
        return result_text, explanation, contributions
    except Exception as e:
        return f"Error: {e}", "An error occurred during prediction.", None

# --- Gradio UI ---
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.header-text { text-align: center; margin-bottom: 2rem; }
"""

with gr.Blocks(title="Space Intelligence Platform") as app:
    gr.Markdown("# 🌌 Space Intelligence Platform", elem_classes=["header-text"])
    gr.Markdown("A dual-pipeline multimodal platform combining image-caption search with planetary transient tracking.", elem_classes=["header-text"])
    
    with gr.Tabs():
        # Tab 1: Vision-Language Model
        with gr.Tab("Deep Space Vision Search (NASA)"):
            with gr.Tabs():
                with gr.Tab("Text-to-Image Search"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            t2i_text_input = gr.Textbox(
                                label="Search Query", 
                                placeholder="e.g., a spiral galaxy, a distant nebula, surface of Mars...",
                                lines=2
                            )
                            t2i_btn = gr.Button("Search Images", variant="primary")
                        with gr.Column(scale=2):
                            t2i_gallery = gr.Gallery(label="Top Matching Images", columns=5)
                    
                    t2i_btn.click(
                        fn=text_to_image_search,
                        inputs=[t2i_text_input],
                        outputs=[t2i_gallery]
                    )
                
                with gr.Tab("Image-to-Image & Captioning"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            i2i_image_input = gr.Image(type="pil", label="Upload Space Imagery")
                            i2i_btn = gr.Button("Analyze & Find Similar", variant="primary")
                        with gr.Column(scale=2):
                            i2i_caption = gr.Markdown("### Best Matching Caption\nUpload an image to see results.")
                            i2i_gallery = gr.Gallery(label="Similar Images from Dataset", columns=5)
                    
                    i2i_btn.click(
                        fn=image_to_image_search,
                        inputs=[i2i_image_input],
                        outputs=[i2i_caption, i2i_gallery]
                    )
            
        # Tab 2: Asteroid Hazard Predictor
        with gr.Tab("Asteroid Hazard Predictor (NASA)"):
            gr.Markdown("### Near-Earth Object Hazard Classification\nPredict whether an asteroid is potentially hazardous based on NASA orbital and physical data.")
            with gr.Row():
                with gr.Column(scale=1):
                    val_mag = gr.Slider(minimum=15.0, maximum=32.0, value=22.0, step=0.1,
                               label="Absolute Magnitude (H) — lower = larger object")
                    val_diam_min = gr.Number(label="Estimated Min Diameter (km)", value=0.1)
                    val_diam_max = gr.Number(label="Estimated Max Diameter (km)", value=0.3)
                    val_vel = gr.Number(label="Relative Velocity (km/h)", value=50000)
                    val_miss = gr.Number(label="Miss Distance (km)", value=5000000)
                    neo_btn = gr.Button("Predict Hazard Status", variant="primary")
                with gr.Column(scale=2):
                    neo_result = gr.Textbox(label="Prediction", lines=3)
                    neo_explanation = gr.Markdown("### Explanation\nEnter asteroid parameters and click predict.")
                    neo_chart = gr.Label(label="Feature Importances")
                    
            neo_btn.click(
                fn=predict_hazard,
                inputs=[val_mag, val_diam_min, val_diam_max, val_vel, val_miss],
                outputs=[neo_result, neo_explanation, neo_chart]
            )

if __name__ == "__main__":
    print("Launching Gradio App...")
    app.launch(share=False, css=custom_css)
