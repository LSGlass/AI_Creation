import torch
from model import TinyGPT, encode, decode

device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model=TinyGPT().to(device)
model.load_state_dict(torch.load('model.pt',map_location=device,weights_only=True)['state_dict'])
model.eval()
print('TinyGPT chat. Type exit to quit.')
while True:
    question=input('You: ').strip()
    if question.lower() in ('exit','quit'): break
    prompt=f'User: {question}\nAssistant:'
    tokens=torch.tensor([encode(prompt)],device=device)
    output=model.generate(tokens,new_tokens=120)
    answer=decode(output[0].tolist()[tokens.size(1):]).split('\nUser:')[0]
    print('AI:',answer)
