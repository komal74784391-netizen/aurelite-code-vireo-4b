"""Dataset preparation utilities for Vireo-4B."""

import json
from pathlib import Path


def prepare(src: str, dst: str, tokenizer: str = "aurelite-vocab-v4"):
    """Tokenize a JSONL corpus and write sharded training files."""
    Path(dst).mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(
        Path(dst) / "train.jsonl", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            record = json.loads(line)
            record["tokenizer"] = tokenizer
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"prepared {count} records into {dst}")


if __name__ == "__main__":
    prepare("data/raw/corpus.jsonl", "data/processed")
