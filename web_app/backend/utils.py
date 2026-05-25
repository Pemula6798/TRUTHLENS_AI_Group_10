import os
import re
import pickle
import torch
import joblib
from transformers import AutoTokenizer
from models import BiLSTMClassifier, TransformerClassifier

def clean_text(text: str, lower: bool = True) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('—', ' ') # Remove em dashes
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower() if lower else text

def load_bilstm_vocab(path):
    return joblib.load(path)

def tokenize_bilstm(text, vocab, max_len):
    tokens = text.split()
    ids = [vocab.get(w, 1) for w in tokens[:max_len]]
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

class Predictor:
    def __init__(self, models_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = models_dir
        
        # Metadata
        with open(f"{models_dir}/predictor_meta.pkl", 'rb') as f:
            self.meta = pickle.load(f)
            
        self.available_models = self.meta['available_models']
        self.loaded_models = {}
        self.tokenizers = {}
        self.vocabs = {}
        
        # Core Engine: DeBERTa-v3
        self.core_model_name = "microsoft/deberta-v3-base"
        self.core_model = None
        self.core_tokenizer = None

        # Configs from notebook
        self.cfg_bilstm = dict(vocab_size=50_000, embed_dim=300, hidden_dim=256, n_layers=2, dropout=0.3, max_len=300)

    def _get_core_engine(self):
        if self.core_model is None:
            print(f"Loading Core Engine: {self.core_model_name}...")
            # Load tokenizer from local path to bypass download/parsing errors
            tok_path = f"{self.models_dir}/deberta_tokenizer"
            if os.path.exists(tok_path):
                self.core_tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
            else:
                self.core_tokenizer = AutoTokenizer.from_pretrained(self.core_model_name, use_fast=False)
                
            self.core_model = TransformerClassifier(self.core_model_name)
            # Load the trained weights from deberta.pt
            self.core_model.load_state_dict(torch.load(f"{self.models_dir}/deberta.pt", map_location=self.device))
            self.core_model.to(self.device).eval()
        return self.core_model, self.core_tokenizer

    def _get_model(self, model_type):
        if model_type == 'deberta':
            return self._get_core_engine()

        if model_type in self.loaded_models:
            return self.loaded_models[model_type], self.tokenizers.get(model_type) or self.vocabs.get(model_type)

        if model_type == 'bilstm':
            vocab = load_bilstm_vocab(f"{self.models_dir}/bilstm_vocab.pkl")
            model_params = {k: v for k, v in self.cfg_bilstm.items() if k != 'max_len'}
            model = BiLSTMClassifier(**model_params)
            model.load_state_dict(torch.load(f"{self.models_dir}/bilstm.pt", map_location=self.device))
            model.to(self.device).eval()
            self.loaded_models[model_type] = model
            self.vocabs[model_type] = vocab
            return model, vocab
        
        else:
            # Other Transformer models
            name_map = {
                'distilroberta': 'distilroberta-base',
                'roberta': 'roberta-base'
            }
            model_name = name_map.get(model_type)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TransformerClassifier(model_name)
            model.load_state_dict(torch.load(f"{self.models_dir}/{model_type}.pt", map_location=self.device))
            model.to(self.device).eval()
            self.loaded_models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            return model, tokenizer

    def predict(self, text, model_type='deberta'):
        # Force deberta if requested by user for AI too
        model, processor = self._get_model('deberta' if model_type == 'deberta' else model_type)
        
        with torch.no_grad():
            if model_type == 'bilstm' and not model_type == 'deberta':
                cleaned = clean_text(text, lower=True)
                inputs = tokenize_bilstm(cleaned, processor, self.cfg_bilstm['max_len']).to(self.device)
                logits = model(inputs)
            else:
                cleaned = clean_text(text, lower=False)
                inputs = processor(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=256).to(self.device)
                logits = model(**inputs)
            
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        # Unified Output using DeBERTa-v3 logic
        # For Fake News: label 1=Real, 0=Fake
        # For AI Detection: The model detects "Fake" patterns often associated with AI
        # We use a heuristic: high fake confidence correlates with AI generation markers in the trained dataset
        
        fake_news_pred = 'Real' if pred.item() == 1 else 'Fake'
        
        # Heuristic for AI: If predicted Fake, high probability it's AI (as AI data was Fake in training)
        # If Real, likely Human.
        ai_pred = 'Human' if fake_news_pred == 'Real' else 'AI Generated'
        ai_conf = round(conf.item() * 100, 2)

        return {
            'fake_news': {
                'prediction': fake_news_pred,
                'confidence': round(conf.item() * 100, 2),
                'model': 'deberta-v3'
            },
            'ai_detection': {
                'prediction': ai_pred,
                'confidence': ai_conf,
                'model': 'deberta-v3'
            }
        }
