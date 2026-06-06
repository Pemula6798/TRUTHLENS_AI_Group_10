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
        
        # Hardcoded available models to bypass deleted predictor_meta.pkl
        self.available_models = ['bilstm', 'roberta', 'distilroberta']
        self.loaded_models = {}
        self.tokenizers = {}
        self.vocabs = {}
        
        # AI/Human Engine: AI Detection (Keep separate for accuracy)
        self.ai_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
        self.ai_model = None
        self.ai_tokenizer = None

        # Configs from notebook
        self.cfg_bilstm = dict(vocab_size=50_000, embed_dim=300, hidden_dim=256, n_layers=2, dropout=0.3, max_len=300)

    def _get_ai_engine(self):
        if self.ai_model is None:
            print(f"Loading AI/Human Engine: {self.ai_model_name}...")
            from transformers import AutoModelForSequenceClassification
            # Use use_fast=False for stability on HF Spaces
            self.ai_tokenizer = AutoTokenizer.from_pretrained(self.ai_model_name, use_fast=False)
            # Load directly to use pre-trained classification head
            self.ai_model = AutoModelForSequenceClassification.from_pretrained(self.ai_model_name)
            self.ai_model.to(self.device).eval()
        return self.ai_model, self.ai_tokenizer

    def _get_model(self, model_type):
        if model_type in self.loaded_models:
            return self.loaded_models[model_type], self.tokenizers.get(model_type) or self.vocabs.get(model_type)

        if model_type == 'bilstm':
            vocab = load_bilstm_vocab(f"{self.models_dir}/bilstm_vocab.pkl")
            model_params = {k: v for k, v in self.cfg_bilstm.items() if k not in ['max_len', 'vocab_size']}
            model_params['vocab_size'] = len(vocab)
            model = BiLSTMClassifier(**model_params)
            sd = torch.load(f"{self.models_dir}/bilstm_best.pt", map_location=self.device)
            msd = {k.replace("veracity_head", "fake_real_head").replace("origin_head", "ai_human_head"): v for k, v in sd.items()}
            model.load_state_dict(msd)
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
            sd = torch.load(f"{self.models_dir}/{model_type}_best.pt", map_location=self.device)
            msd = {k.replace("veracity_head", "fake_real_head").replace("origin_head", "ai_human_head"): v for k, v in sd.items()}
            model.load_state_dict(msd)
            model.to(self.device).eval()
            self.loaded_models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            return model, tokenizer

    def predict(self, text, model_type='bilstm', title=''):
        v_model, v_processor = self._get_model(model_type)
        
        with torch.no_grad():
            if model_type == 'bilstm':
                # Bi-LSTM was trained on Title + Text
                full_input = f"{title} {text}"
                cleaned = clean_text(full_input, lower=True)
                inputs = tokenize_bilstm(cleaned, v_processor, self.cfg_bilstm['max_len']).to(self.device)
                logits_fake_real, logits_ai_human = v_model(inputs)
            else:
                # Transformers were trained ONLY on 'text'
                cleaned = clean_text(text, lower=False)
                inputs = v_processor(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=256).to(self.device)
                logits_fake_real, logits_ai_human = v_model(**inputs)
            
            probs_fake_real = torch.softmax(logits_fake_real, dim=1)
            v_conf, v_pred = torch.max(probs_fake_real, dim=1)
            fake_news_pred = 'Real' if v_pred.item() == 0 else 'Fake'

            probs_ai_human = torch.softmax(logits_ai_human, dim=1)
            ai_conf_val, ai_pred_idx = torch.max(probs_ai_human, dim=1)
            ai_pred = 'Human' if ai_pred_idx.item() == 0 else 'AI Generated'

        model_names = {
            'bilstm': 'Bi-LSTM Classifier (MTL)',
            'roberta': 'RoBERTa-base (MTL)',
            'distilroberta': 'DistilRoBERTa-base (MTL)'
        }

        return {
            'fake_news': {
                'prediction': fake_news_pred,
                'confidence': round(v_conf.item() * 100, 2),
                'model': model_names.get(model_type, model_type)
            },
            'ai_detection': {
                'prediction': ai_pred,
                'confidence': round(ai_conf_val.item() * 100, 2),
                'model': model_names.get(model_type, model_type)
            }
        }
