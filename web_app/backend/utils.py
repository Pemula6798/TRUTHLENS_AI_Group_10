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
        
        # Origin Engine: AI Detection (Keep separate for accuracy)
        self.ai_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
        self.ai_model = None
        self.ai_tokenizer = None

        # Configs from notebook
        self.cfg_bilstm = dict(vocab_size=50_000, embed_dim=300, hidden_dim=256, n_layers=2, dropout=0.3, max_len=300)

    def _get_ai_engine(self):
        if self.ai_model is None:
            print(f"Loading Origin Engine: {self.ai_model_name}...")
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
            model_params = {k: v for k, v in self.cfg_bilstm.items() if k != 'max_len'}
            model = BiLSTMClassifier(**model_params)
            model.load_state_dict(torch.load(f"{self.models_dir}/bilstm_best.pt", map_location=self.device))
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
            model.load_state_dict(torch.load(f"{self.models_dir}/{model_type}_best.pt", map_location=self.device))
            model.to(self.device).eval()
            self.loaded_models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            return model, tokenizer

    def predict(self, text, model_type='bilstm', title=''):
        # 1. Veracity Prediction (Fake/Real)
        v_model, v_processor = self._get_model(model_type)
        
        with torch.no_grad():
            if model_type == 'bilstm':
                # Bi-LSTM was trained on Title + Text
                full_input = f"{title} {text}"
                cleaned = clean_text(full_input, lower=True)
                inputs = tokenize_bilstm(cleaned, v_processor, self.cfg_bilstm['max_len']).to(self.device)
                logits = v_model(inputs)
            else:
                # IMPORTANT: DeBERTa/RoBERTa were trained ONLY on 'text' in the notebook.
                # Adding title here was causing the poor accuracy.
                cleaned = clean_text(text, lower=False)
                inputs = v_processor(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=256).to(self.device)
                logits = v_model(**inputs)
            
            probs = torch.softmax(logits, dim=1)
            v_conf, v_pred = torch.max(probs, dim=1)
            # Dataset convention from WELFake: 0=Real, 1=Fake
            # The model learned: class 0 → Real, class 1 → Fake
            fake_news_pred = 'Real' if v_pred.item() == 0 else 'Fake'

        # 2. Origin Prediction (Human/AI)
        ai_model, ai_processor = self._get_ai_engine()
        with torch.no_grad():
            ai_cleaned = clean_text(text, lower=False)
            ai_inputs = ai_processor(ai_cleaned, return_tensors='pt', truncation=True, padding=True, max_length=256).to(self.device)
            # Use AutoModel directly, so we need .logits
            ai_outputs = ai_model(**ai_inputs)
            ai_probs = torch.softmax(ai_outputs.logits, dim=1)
            ai_conf_val, ai_pred_idx = torch.max(ai_probs, dim=1)
            ai_pred = 'Human' if ai_pred_idx.item() == 0 else 'AI Generated'

        model_names = {
            'bilstm': 'Bi-LSTM Classifier',
            'roberta': 'RoBERTa-base',
            'distilroberta': 'DistilRoBERTa-base'
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
                'model': 'chatgpt-detector'
            }
        }
