# 🌌 Space Image Caption Generator

An AI system that automatically generates descriptive captions for space images using deep learning. The model combines **Computer Vision** and **Natural Language Processing** to understand images and convert them into meaningful sentences.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Problem Type** | Image Captioning |
| **Encoder (Baseline)** | CNN (ResNet50) |
| **Encoder (Experiment)** | Vision Transformer (ViT) |
| **Decoder (Baseline)** | LSTM |
| **Decoder (Experiment)** | Transformer |
| **Evaluation Metric** | BLEU Score |
| **Demo Framework** | Gradio |

---

## 🗂️ Project Structure

```
space_caption_generator/
│
├── dataset/
│   ├── images/                  # All scraped space images
│   └── captions.csv             # Cleaned image-caption pairs
│
├── scraping/
│   ├── nasa_scraper.py          # Scrapes images from NASA API
│   └── clean_captions.py        # Cleans and simplifies raw captions
│
├── preprocessing/
│   ├── image_prep.py            # Resize, normalize images (224x224)
│   └── text_prep.py             # Tokenization, vocabulary builder
│
├── models/
│   ├── cnn_encoder.py           # ResNet50-based image encoder
│   ├── vit_encoder.py           # Vision Transformer encoder
│   ├── lstm_decoder.py          # LSTM-based caption decoder
│   ├── transformer_decoder.py   # Transformer-based caption decoder
│   └── baseline_model.py        # Combines encoder + decoder
│
├── training/
│   ├── train_baseline.py        # Train CNN + LSTM
│   ├── train_vit.py             # Train ViT + LSTM
│   └── train_transformer.py     # Train ViT + Transformer
│
├── experiments/
│   └── results.csv              # BLEU scores for all model configs
│
├── demo/
│   └── app.py                   # Gradio web demo
│
├── config.py                    # Hyperparameters and paths
├── utils.py                     # Shared utility functions
└── README.md
```

---

## ⚙️ Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-username/space-caption-generator.git
cd space-caption-generator

# Install dependencies
pip install torch torchvision gradio pillow pandas nltk requests beautifulsoup4 transformers timm anthropic
```

---

## 🚀 How to Run

### 1. Collect Data
```bash
python scraping/nasa_scraper.py
python scraping/clean_captions.py
```

### 2. Train Baseline Model (CNN + LSTM)
```bash
python training/train_baseline.py
```

### 3. Run Ablation Experiments
```bash
python training/train_vit.py
python training/train_transformer.py
```

### 4. Evaluate Models
```bash
python experiments/evaluate.py
```

### 5. Launch Demo
```bash
python demo/app.py
```

---

## 🧠 Model Architecture

### Full Pipeline

```
Image → CNN/ViT Encoder → Feature Vector → LSTM/Transformer Decoder → Caption
```

### Encoder
- **Baseline:** ResNet50 (pretrained on ImageNet) — outputs a 2048-dim feature vector
- **Experiment:** ViT-base-patch16-224 — uses CLS token as image representation

### Decoder
- **Baseline:** LSTM — generates words one at a time from the feature vector
- **Experiment:** Transformer — uses self-attention for better long-range dependencies

---

## 📊 Experiment Results

| Model Config | BLEU-1 | BLEU-2 | BLEU-4 |
|---|---|---|---|
| CNN + LSTM (Baseline) | 0.42 | 0.28 | 0.12 |
| ViT + LSTM | 0.51 | 0.35 | 0.18 |
| CNN + Transformer | 0.48 | 0.33 | 0.16 |
| ViT + Transformer | **0.55** | **0.39** | **0.21** |

> Best performance achieved with **ViT + Transformer** configuration.

---

## 🗃️ Dataset

- **Source:** NASA Image and Video Library API
- **Size:** ~500 images
- **Caption Style:** Short, descriptive, 5–10 words
- **Format:**

```
image,caption
galaxy_0.jpg,a spiral galaxy in deep space
nebula_1.jpg,colorful nebula with bright gas clouds
planet_2.jpg,a red rocky planet surface
```

> Captions were collected from NASA metadata and simplified using an AI cleaning pipeline to ensure consistency and suitability for training.

---

## 🖥️ Demo

Upload any space image and the model will generate a caption automatically.

```bash
python demo/app.py
```

A public Gradio link will be printed in the terminal — shareable with anyone.

---

## 📁 Configuration

All hyperparameters are in `config.py`:

```python
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
MAX_CAPTION_LEN = 25
FREQ_THRESHOLD = 2
IMG_SIZE = 224
```

---

## 👥 Team

| Alaa orabi |
| lamis abdallah | 
| fatma sameh | 
| rawan essam | 
| shrouk wael| 

---

## 📄 Report Summary

> *"We built an AI system that generates descriptive captions for space images using deep learning techniques combining computer vision and natural language processing. We collected image captions from NASA's public API and manually refined them to ensure consistency and suitability for training."*

---

