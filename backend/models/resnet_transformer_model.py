import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights
import math
import re
import pandas as pd
from collections import Counter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def tokenize(t):
    return t.split()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(2048, 256)

        for name, p in self.backbone.named_parameters():
            p.requires_grad = "layer4" in name

    def forward(self, x):
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        feat = feat.permute(0,2,3,1)
        feat = feat.view(B, H*W, C)
        feat = self.proj(feat)
        return feat

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 256)
        self.pos   = PositionalEncoding(256)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8,
                dim_feedforward=512,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=3
        )
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, memory, caps):
        x = self.embed(caps)
        x = self.pos(x)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            caps.size(1), device=caps.device
        )
        out = self.transformer(x, memory, tgt_mask=tgt_mask)
        return self.fc(out)


class SpaceResnetTransformerModel:
    def __init__(self):
        self.device = device
        self.vocab = None
        self.encoder = None
        self.decoder = None
        
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        
        self._load_model()

    def _load_model(self):
        # We assume the weights are saved in backend/models/best_model.pth
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        
        if os.path.exists(model_path):
            print(f"Loading ResNet50+Transformer from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.vocab = checkpoint['vocab']
            
            vocab_size = len(self.vocab)
            self.encoder = EncoderResNet().to(self.device)
            self.decoder = Decoder(vocab_size).to(self.device)
            
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
        else:
            print(f"WARNING: '{model_path}' not found! Initializing with random weights.")
            # Build vocabulary dynamically to prevent crashes
            self.vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
            dataset_csv = os.path.join(os.path.dirname(__file__), "..", "dataset", "vlm_captions_dataset.csv")
            if os.path.exists(dataset_csv):
                df = pd.read_csv(dataset_csv)
                counter = Counter()
                for c in df["caption"]:
                    words = tokenize(clean(c))
                    counter.update(words)
                for w, f in counter.items():
                    if f >= 3:
                        self.vocab[w] = len(self.vocab)
            else:
                # Minimal fallback if csv is not found
                self.vocab["galaxy"] = 4
                self.vocab["stars"] = 5
            
            vocab_size = len(self.vocab)
            self.encoder = EncoderResNet().to(self.device)
            self.decoder = Decoder(vocab_size).to(self.device)
            
        self.encoder.eval()
        self.decoder.eval()

    def generate_caption(self, image, beam_size=3, max_len=80):
        img_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            memory    = self.encoder(img_tensor.unsqueeze(0))
            beams     = [(0.0, [self.vocab["<START>"]])]
            completed = []

            for _ in range(max_len):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == self.vocab["<END>"]:
                        completed.append((score, seq))
                        continue
                    inp       = torch.tensor(seq).unsqueeze(0).to(self.device)
                    out       = self.decoder(memory, inp)
                    log_probs = torch.log_softmax(out[0, -1], dim=0)

                    # small length penalty/repetition penalty
                    for prev_idx in seq[-8:]:
                        if prev_idx > 3:
                            log_probs[prev_idx] -= 2.5

                    topk      = log_probs.topk(beam_size)
                    for log_p, idx in zip(topk.values, topk.indices):
                        candidates.append((score + log_p.item(),
                                           seq + [idx.item()]))
                
                # Keep top beam_size
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            best = max(completed if completed else beams, key=lambda x: x[0])

        rev = {v: k for k, v in self.vocab.items()}
        caption_words = [rev[i] for i in best[1][1:] if i in rev and i > 3 and i != self.vocab["<END>"]]
        
        return " ".join(caption_words)
