import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Bi-LSTM Components
class AttentionPool(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w = nn.Linear(h * 2, 1)

    def forward(self, x):
        # x: (batch, seq, hidden*2)
        attn_weights = torch.softmax(self.w(x), dim=1)
        return (x * attn_weights).sum(1)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attn = AttentionPool(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        e, _ = self.lstm(self.drop(self.emb(x)))
        return self.fc(self.drop(self.attn(e)))

# Transformer Wrapper
class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, **kwargs):
        out = self.model(**kwargs)
        return out.logits
