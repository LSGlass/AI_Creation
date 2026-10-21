import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width), nn.Dropout(dropout))
    def forward(self, x):
        t = x.size(1)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + y
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab=256, context=128, width=128, layers=4, heads=4):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
    def forward(self, ids, targets=None):
        b,t = ids.shape
        if t > self.context: raise ValueError('Sequence exceeds context window')
        x = self.token(ids) + self.pos(torch.arange(t, device=ids.device))
        logits = self.head(self.norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, ids, new_tokens=120, temperature=0.8, top_k=30):
        self.eval()
        for _ in range(new_tokens):
            logits,_ = self(ids[:, -self.context:])
            logits = logits[:, -1, :] / max(temperature, 0.01)
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
        return ids

def encode(text): return list(text.encode('utf-8'))
def decode(ids): return bytes(ids).decode('utf-8', errors='replace')
