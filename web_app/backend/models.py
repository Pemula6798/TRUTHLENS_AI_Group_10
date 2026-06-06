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
        self.pool = AttentionPool(hidden_dim)
        self.fake_real_head = nn.Linear(hidden_dim * 2, 2)
        self.ai_human_head   = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb_out = self.emb(x)
        lstm_out, _ = self.lstm(emb_out)
        pooled = self.pool(lstm_out)
        logits_fake_real = self.fake_real_head(pooled)
        logits_ai_human = self.ai_human_head(pooled)
        return logits_fake_real, logits_ai_human

# Transformer Wrapper
class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        h_size = self.encoder.config.hidden_size
        self.fake_real_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(h_size, h_size), nn.Tanh(),
            nn.Dropout(0.1), nn.Linear(h_size, 2)
        )
        self.ai_human_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(h_size, h_size), nn.Tanh(),
            nn.Dropout(0.1), nn.Linear(h_size, 2)
        )

    def forward(self, **kwargs):
        # RoBERTa and DistilRoBERTa do not use token_type_ids
        if 'token_type_ids' in kwargs and self.encoder.config.model_type in ['roberta', 'distilroberta']:
            kwargs.pop('token_type_ids', None)
        out = self.encoder(**kwargs)
        cls_rep = out.last_hidden_state[:, 0, :]
        
        # Cast cls_rep to match head parameter dtype to prevent half/float mismatch
        head_dtype = next(self.fake_real_head.parameters()).dtype
        cls_rep = cls_rep.to(head_dtype)
        
        return self.fake_real_head(cls_rep), self.ai_human_head(cls_rep)

