import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from safetensors.torch import load_file
import os


class SpaceCaptionModel:

    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning"):
        print(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load local safetensors weights
        local_weights = os.path.join(os.path.dirname(__file__), "model.safetensors")
        if os.path.exists(local_weights):
            print(f"Loading local trained weights from {local_weights}...")
            state_dict = load_file(local_weights)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"WARNING: {local_weights} not found, using default HuggingFace weights.")

        # Set special tokens for the decoder as required by the model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        
        # Generation parameters tuned for space imagery descriptions
        self.gen_kwargs = {
            "max_length": 50,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "no_repeat_ngram_size": 2,
        }
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        """
        Takes a PIL image and returns an AI-generated string caption.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption


print("ViT-GPT2 Caption Generator Ready!")
