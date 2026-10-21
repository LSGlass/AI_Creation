import argparse

import torch

from model import TinyGPT, encode, decode

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="model.pt",
    help="Path to the trained model checkpoint"
)

args = parser.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

checkpoint = torch.load(
    args.model,
    map_location=device,
    weights_only=True
)

model = TinyGPT(
    context=checkpoint.get("context", 128)
).to(device)

model.load_state_dict(checkpoint["state_dict"])

model.eval()

print(f"Loaded model: {args.model}")
print("TinyGPT chat. Type exit to quit.")

while True:

    question = input("You: ").strip()

    if question.lower() in ("exit", "quit"):
        break

    if not question:
        continue

    prompt = f"User: {question}\nAssistant:"

    tokens = torch.tensor(
        [encode(prompt)],
        device=device
    )

    with torch.inference_mode():
        output = model.generate(
            tokens,
            new_tokens=120,
            temperature=0.5,
            top_k=10
        )

    answer = decode(
        output[0].tolist()[tokens.size(1):]
    ).split("\nUser:")[0]

    print("AI:", answer)
