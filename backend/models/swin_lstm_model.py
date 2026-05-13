import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image

from transformers import AutoImageProcessor, SwinModel

# Recreating the Keras Tokenizer and pad_sequences to avoid importing heavy TensorFlow
class SimpleTokenizer:
    def __init__(self, oov_token="<unk>"):
        self.oov_token = oov_token
        self.word_index = {}
        # Keras Tokenizer typically leaves 0 as a reserved padding token
        self.word_index[oov_token] = 1 
        
    def fit_on_texts(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            words = str(text).lower().split()
            counter.update(words)
        
        # Sort by frequency like Keras
        for word, _ in counter.most_common():
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index) + 1
                
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = str(text).lower().split()
            seq = [self.word_index.get(w, self.word_index[self.oov_token]) for w in words]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen, padding="post"):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            if padding == "post":
                seq = seq + [0] * (maxlen - len(seq))
            else:
                seq = [0] * (maxlen - len(seq)) + seq
        padded.append(seq)
    return padded

class Decoder(nn.Module):
    def __init__(self, vocab_size, feature_dim, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.feature_layer = nn.Linear(feature_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        img_emb = self.feature_layer(features).unsqueeze(1)
        embeddings = torch.cat((img_emb, embeddings[:, :-1, :]), dim=1)
        outputs, _ = self.lstm(embeddings)
        return self.fc(outputs)


class SpaceSwinLSTMModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Swin Encoder
        print("Loading Swin Encoder...")
        model_id = "microsoft/swin-tiny-patch4-window7-224"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.encoder = SwinModel.from_pretrained(model_id).to(self.device)
        self.encoder.eval()
        
        self.hidden_size = self.encoder.config.hidden_size
        
        # Load Vocabulary
        dataset_csv = os.path.join(os.path.dirname(__file__), "..", "dataset", "vlm_captions_dataset.csv")
        self.tokenizer = SimpleTokenizer(oov_token="<unk>")
        
        if os.path.exists(dataset_csv):
            df = pd.read_csv(dataset_csv)
            df["text"] = df["title"].astype(str) + " : " + df["caption"].astype(str)
            df["text"] = "<start> " + df["text"].str.lower() + " <end>"
            self.tokenizer.fit_on_texts(df["text"])
            
            # Compute max length
            sequences = self.tokenizer.texts_to_sequences(df["text"])
            self.max_length = max(len(seq) for seq in sequences)
        else:
            self.tokenizer.fit_on_texts(["<start> space <end>"])
            self.max_length = 50
            
        vocab_size = len(self.tokenizer.word_index) + 1
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        
        # Build Decoder
        self.decoder = Decoder(vocab_size, self.hidden_size).to(self.device)
        
        # Load Decoder weights
        weights_path = os.path.join(os.path.dirname(__file__), "swin_lstm_decoder.pth")
        if os.path.exists(weights_path):
            print(f"Loading Swin-LSTM Decoder weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.decoder.load_state_dict(state_dict)
        else:
            print(f"WARNING: {weights_path} not found. Using untrained LSTM decoder weights.")
            
        self.decoder.eval()

    def generate_caption(self, image: Image.Image, beam_size=5) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            
        features = outputs.last_hidden_state.mean(dim=1)
        
        start_idx = self.tokenizer.word_index.get("<start>", 1)
        end_idx = self.tokenizer.word_index.get("<end>", 2)
        
        beams = [([start_idx], 0.0)]
        
        for _ in range(self.max_length):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == end_idx:
                    new_beams.append((seq, score))
                    continue
                    
                cap_input = pad_sequences([seq], maxlen=self.max_length, padding="post")
                cap_input = torch.tensor(cap_input, dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    preds = self.decoder(features, cap_input)
                    
                pos = len(seq) - 1
                probs = F.softmax(preds[0, pos], dim=0)
                
                top_probs, top_words = torch.topk(probs, beam_size)
                
                for p, w in zip(top_probs, top_words):
                    new_beams.append((seq + [w.item()], score + torch.log(p).item()))
                    
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if all(b[0][-1] == end_idx for b in beams):
                break
                
        best = beams[0][0]
        sentence = []
        for i in best:
            w = self.index_word.get(i, "")
            if w not in ["<start>", "<end>", ""]:
                sentence.append(w)
                
        return " ".join(sentence)
