"""Training entry point for Vireo-4B."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .model import Vireo4bModel


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = Vireo4bModel(**cfg["model"])
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]))

    losses = []
    for step in range(args.steps):
        tokens = torch.randint(0, 32000, (cfg["data"]["batch_size"], cfg["data"]["seq_len"]))
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss))
        if step % 100 == 0:
            print(f"step={step} loss={losses[-1]:.4f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/train_loss.json").write_text(json.dumps(losses[-10:]), encoding="utf-8")


if __name__ == "__main__":
    main()
