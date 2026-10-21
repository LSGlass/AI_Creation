import streamlit as st
import torch
from model import TinyGPT, encode, decode

st.title('My Scratch AI Assistant')
@st.cache_resource
def load():
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model=TinyGPT().to(device)
    model.load_state_dict(torch.load('model.pt',map_location=device,weights_only=True)['state_dict'])
    model.eval()
    return model,device
try:
    model,device=load()
except FileNotFoundError:
    st.error('Run python train.py first to create model.pt')
    st.stop()
if 'messages' not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']): st.write(m['content'])
if question:=st.chat_input('Ask your AI something'):
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'): st.write(question)
    prompt=f'User: {question}\nAssistant:'
    ids=torch.tensor([encode(prompt)],device=device)
    output=model.generate(ids,new_tokens=120)
    answer=decode(output[0].tolist()[ids.size(1):]).split('\nUser:')[0]
    with st.chat_message('assistant'): st.write(answer)
    st.session_state.messages.append({'role':'assistant','content':answer})
