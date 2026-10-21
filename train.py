import argparse
from pathlib import Path
import torch
from model import TinyGPT, encode

p=argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=600)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--data', default='data.txt')
a=p.parse_args()
torch.manual_seed(42)
device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
raw=Path(a.data).read_text(encoding='utf-8')
ids=torch.tensor(encode(raw), dtype=torch.long)
context=128
if len(ids) < context+2: raise SystemExit('Training data must be at least 130 UTF-8 bytes')
cut=int(len(ids)*0.9)
train=ids[:cut]; val=ids[cut:]
if len(train) <= context: raise SystemExit('Add more training data')
model=TinyGPT(context=context).to(device)
opt=torch.optim.AdamW(model.parameters(), lr=a.lr)

def batch(source):
    ix=torch.randint(0,len(source)-context,(a.batch,))
    x=torch.stack([source[i:i+context] for i in ix]).to(device)
    y=torch.stack([source[i+1:i+context+1] for i in ix]).to(device)
    return x,y
for step in range(1,a.steps+1):
    model.train()
    x,y=batch(train)
    _,loss=model(x,y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step()
    if step==1 or step%50==0:
        print(f'step {step}/{a.steps} | training loss {loss.item():.4f}',flush=True)
torch.save({'state_dict':model.cpu().state_dict()}, 'model.pt')
print('Saved model.pt')
