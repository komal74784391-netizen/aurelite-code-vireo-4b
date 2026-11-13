# Vireo-4B — Official Code Release

**Accepted at NeurIPS 2025.** Official training, evaluation and inference code for Vireo-4B.

This repository contains the complete training, evaluation and inference code that was
used to produce the results reported in the paper. Weights are released under Apache-2.0.

## Repository layout

- `src/model.py` — reference model implementation
- `src/train.py` — training loop and config loader
- `data/prepare.py` — corpus preparation utilities
- `experiments/base.yaml` — reference training configuration
- `notebooks/quickstart.ipynb` — end-to-end inference demo
- `results/metrics.json` — benchmark numbers from the paper
- `tests/test_smoke.py` — smoke tests
- `requirements.txt` — runtime dependencies

## Quick start

```bash
pip install -r requirements.txt
python -m src.train --config experiments/base.yaml --steps 200
```

## Reported metrics

```json
{
  "arc_challenge": 62.1,
  "gsm8k": 71.4,
  "math500": 38.9
}
```

## Citation

```bibtex
@inproceedings{aurelite2025vireo4b,
  title={Vireo-4B: Efficient Reasoning at Small Scale},
  author={Aurelite Labs},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
