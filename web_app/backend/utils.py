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
        
        # Core Engine: Bi-LSTM (The most accurate in training: 96%)
        self.core_model_name = "bilstm"
        self.core_model = None
        self.core_tokenizer = None

        # Configs from notebook
        self.cfg_bilstm = dict(vocab_size=50_000, embed_dim=300, hidden_dim=256, n_layers=2, dropout=0.3, max_len=300)

    def _get_core_engine(self):
        if self.core_model is None:
            print(f"Loading Veracity Engine: {self.core_model_name}...")
            # Load Bi-LSTM components
            vocab = load_bilstm_vocab(f"{self.models_dir}/bilstm_vocab.pkl")
            model_params = {k: v for k, v in self.cfg_bilstm.items() if k != 'max_len'}
            model = BiLSTMClassifier(**model_params)
            model.load_state_dict(torch.load(f"{self.models_dir}/bilstm.pt", map_location=self.device))
            model.to(self.device).eval()
            self.core_model = model
            self.core_tokenizer = vocab
        return self.core_model, self.core_tokenizer

    def _get_ai_engine(self):
        if self.ai_model is None:
            print(f"Loading Origin Engine: {self.ai_model_name}...")
            # Use use_fast=False for stability on HF Spaces
            self.ai_tokenizer = AutoTokenizer.from_pretrained(self.ai_model_name, use_fast=False)
            self.ai_model = TransformerClassifier(self.ai_model_name)
            self.ai_model.to(self.device).eval()
        return self.ai_model, self.ai_tokenizer

    def _get_model(self, model_type):
        if model_type == 'bilstm' or model_type == 'deberta': # Use Bi-LSTM as core for both if requested for stability
            return self._get_core_engine()

        if model_type in self.loaded_models:
            return self.loaded_models[model_type], self.tokenizers.get(model_type) or self.vocabs.get(model_type)
        
        # ... fallback for others
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

    def predict(self, text, model_type='deberta', title=''):
        # 1. Veracity Prediction (Fake/Real)
        # Note: In notebook, Bi-LSTM was trained on Title + Text, Transformers only on Text.
        
        v_model, v_processor = self._get_model('bilstm') # Force Bi-LSTM for 96% accuracy
        
        with torch.no_grad():
            # Bi-LSTM expects combined input
            full_input = f"{title} {text}"
            cleaned = clean_text(full_input, lower=True)
            inputs = tokenize_bilstm(cleaned, v_processor, self.cfg_bilstm['max_len']).to(self.device)
            logits = v_model(inputs)
            
            probs = torch.softmax(logits, dim=1)
            v_conf, v_pred = torch.max(probs, dim=1)
            fake_news_pred = 'Real' if v_pred.item() == 1 else 'Fake'

        # 2. Origin Prediction (Human/AI)
        ai_model, ai_processor = self._get_ai_engine()
        with torch.no_grad():
            ai_cleaned = clean_text(text, lower=False)
            ai_inputs = ai_processor(ai_cleaned, return_tensors='pt', truncation=True, padding=True, max_length=256).to(self.device)
            ai_logits = ai_model(**ai_inputs)
            ai_probs = torch.softmax(ai_logits, dim=1)
            ai_conf_val, ai_pred_idx = torch.max(ai_probs, dim=1)
            ai_pred = 'Human' if ai_pred_idx.item() == 0 else 'AI Generated'

        return {
            'fake_news': {
                'prediction': fake_news_pred,
                'confidence': round(v_conf.item() * 100, 2),
                'model': 'Bi-LSTM (Core)'
            },
            'ai_detection': {
                'prediction': ai_pred,
                'confidence': round(ai_conf_val.item() * 100, 2),
                'model': 'chatgpt-detector'
            }
        }
