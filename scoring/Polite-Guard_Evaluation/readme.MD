# Polite-Guard Evaluation

This folder contains evaluation results using the HuggingFace model [Intel/polite-guard](https://huggingface.co/Intel/polite-guard).

## Contents
- `polite_guard_eval.py`  
  Script to run polite-guard on model outputs (JSON). Produces per-response politeness scores in [-1, 1].

- `scores_polite_guard-8B.csv`  
  Politeness scores for LLaMA3.1-8B baseline and pruned layers.

- `scores_polite_guard-1B.csv`  
  Politeness scores for LLaMA3.2-1B baseline and pruned layers.

- `Average_score_per_model-8B.csv`  
  Average scores per model configuration (8B).

- `Average_score_per_model-1B.csv`  
  Average scores per model configuration (1B).

## Usage
Run polite-guard scoring:
```bash
python polite_guard_eval.py --input_dir LLaMA3.1-8B --out scores_polite_guard-8B.csv
python polite_guard_eval.py --input_dir LLaMA3.2-1B --out scores_polite_guard-1B.csv
