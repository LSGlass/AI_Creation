# Scratch AI Assistant

A small decoder-only Transformer trained from random initialization. This is an educational model, not a ChatGPT-quality assistant.

## Setup

```bash
python -m venv .venv
```

Activate: Windows PowerShell `.venv\Scripts\Activate.ps1`; macOS/Linux `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
python train.py --steps 600
python chat.py
streamlit run app.py
```

## Files
- `model.py`: causal Transformer and byte tokenizer
- `train.py`: next-token training loop
- `data.txt`: small synthetic demonstration corpus
- `chat.py`: terminal chat
- `app.py`: browser chat UI

## Limitations
The supplied data is tiny, repetitive, and not representative. Training loss is not a measure of conversation quality. The model will mostly memorize patterns and generate incoherent text. It has a 128-byte context and no instruction-tuning or long-term memory. To improve it, train on a large, legally usable, diverse corpus, use a stronger tokenizer, hold out evaluation data, and then instruction-tune on conversations. CPU training may be slow.
