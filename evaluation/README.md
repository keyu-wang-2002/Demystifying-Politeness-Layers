
# 📘 Evaluation Module

## 📌 Overview

This folder (`evaluation/`) contains all scripts and results for
**politeness evaluation**.\
We implement **three complementary approaches**:

1.  **GPT Continuous Scoring** (`gpt_eval/`)
    -   Assigns a politeness score on a continuous scale **\[-1, 1\]**.\
    -   Captures nuanced differences in politeness tone.
2.  **Frequency-based Scoring** (`frequency/`)
    -   Counts **polite markers** (e.g., *please, thank you, sorry*).\
    -   Reports word- and sentence-level frequency metrics.
3.  **Polite-Guard Evaluation** (`polite_guard_eval/`)
    -   Uses HuggingFace's **Polite-Guard** model.\
    -   Provides **layer-wise politeness scoring** for deeper analysis.

------------------------------------------------------------------------

## 📂 Structure

    evaluation/
    │
    ├── gpt_eval/             # Continuous scoring with GPT
    │   └── README.md
    │
    ├── frequency/            # Frequency-based scoring
    │   └── README.md
    │
    ├── polite_guard_eval/    # HuggingFace Polite-Guard evaluation
    │   └── README.md
    │
    └── README.md             # This file (summary of evaluation methods)

------------------------------------------------------------------------

## ▶️ How to Use

Each subfolder contains: - Scripts (`*.py`) for running evaluations\
- Result files (`scores-*.csv`, `Average_score_*.csv`)\
- A local `README.md` with detailed instructions

### Example commands

**GPT Continuous Scoring**

``` bash
python gpt_eval/politeness_score_batch_chunks_resume_logged.py     --input_dir ../data/     --out gpt_eval/scores-1B.csv     --model gpt-4o-mini
```

**Frequency Scoring**

``` bash
python frequency/gpt_politeword_freq.py     --input_dir ../data/     --out frequency/scores-1B-sentence.csv     --model gpt-4o-mini
```

**Polite-Guard**

``` bash
python polite_guard_eval/run_eval.py     --input_dir ../data/     --out polite_guard_eval/scores-1B.csv
```

------------------------------------------------------------------------

## 📑 Outputs

-   **Raw scores**: `scores-*.csv`\
-   **Aggregated scores**: `Average_score_*.csv`\
-   **Raw model outputs**: stored in `raw_outputs/` (for GPT evaluators)

------------------------------------------------------------------------

## 🏆 Notes

-   The three methods are **complementary**:
    -   GPT scoring: nuanced continuous judgments\
    -   Frequency scoring: interpretable, rule-based metric\
    -   Polite-Guard: external HF classifier for cross-validation
-   All results follow a **standardized CSV format**, making them easy
    to compare.
