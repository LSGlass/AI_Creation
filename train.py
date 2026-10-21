import argparse
from pathlib import Path

import torch

from model import TinyGPT, encode

parser = argparse.ArgumentParser()

parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--data", default="data_v2.txt")
parser.add_argument("--output", default="model_v2.pt")

args = parser.parse_args()

torch.manual_seed(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Training device: {device}")

# Load conversation examples
raw = Path(args.data).read_text(encoding="utf-8")

examples = [
    example.strip()
    for example in raw.split("\n\n")
    if example.strip()
]

# Remove duplicate conversations
examples = list(dict.fromkeys(examples))

print(f"Unique conversations: {len(examples)}")

if len(examples) < 10:
    raise SystemExit("Dataset needs at least 10 unique conversations.")

# Shuffle examples before splitting
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(examples), generator=generator).tolist()

shuffled = [examples[i] for i in indices]

split = int(len(shuffled) * 0.9)

train_examples = shuffled[:split]
val_examples = shuffled[split:]

print(f"Training conversations: {len(train_examples)}")
print(f"Validation conversations: {len(val_examples)}")

# Convert conversations into token sequences
train_text = "\n\n".join(train_examples) + "\n"
val_text = "\n\n".join(val_examples) + "\n"

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)

context = 128

if len(train_ids) <= context or len(val_ids) <= context:
    raise SystemExit("Not enough training or validation data.")

model = TinyGPT(context=context).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr
)

def get_batch(source):
    positions = torch.randint(
        0,
        len(source) - context,
        (args.batch,)
    )

    x = torch.stack([
        source[i:i + context]
        for i in positions
    ]).to(device)

    y = torch.stack([
        source[i + 1:i + context + 1]
        for i in positions
    ]).to(device)

    return x, y


@torch.no_grad()
def evaluate():
    model.eval()

    losses = []

    for _ in range(10):
        x, y = get_batch(val_ids)

        _, loss = model(x, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


print("Starting training...")

for step in range(1, args.steps + 1):

    model.train()

    x, y = get_batch(train_ids)

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if step == 1 or step % 50 == 0:

        validation_loss = evaluate()

        print(
            f"Step {step}/{args.steps} | "
            f"Training Loss: {loss.item():.4f} | "
            f"Validation Loss: {validation_loss:.4f}",
            flush=True
        )

torch.save(
    {
        "state_dict": model.cpu().state_dict(),
        "context": context,
        "training_steps": args.steps
    },
    args.output
)

print(f"Saved {args.output}")