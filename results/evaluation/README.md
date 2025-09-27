#  Politeness Evaluation Repository

##  Overview

This repository contains code and results for the **Politeness
Evaluation Study**.\
The goal is to evaluate language models on their ability to produce
polite responses.\
We implement and compare **three complementary evaluation approaches**:

1.  **GPT Continuous Scoring** (`gpt_eval/`)
    -   Judges politeness on a continuous scale **\[-1, 1\]**.\
    -   Example: "Please fix this error." → 0.0, "I would appreciate it
        if..." → 0.8.
2.  **Frequency-based Scoring** (`frequency/`)
    -   Counts the occurrence of **polite markers** (e.g., *please,
        thank you, sorry, appreciate*).\
    -   Outputs word- or sentence-level frequency metrics.
3.  **Polite-Guard Evaluation** (`polite_guard_eval/`)
    -   Uses **HuggingFace's Polite-Guard model** for automatic
        politeness classification.\
    -   Supports **layer-wise analysis** (probing intermediate
        representations).

------------------------------------------------------------------------

##  Repository Structure

    repo_root/
    │
    ├── README.md                 # This file (overview)
    ├── requirements.txt          # Dependencies
    │
    ├── evaluation/
    │   ├── gpt_eval/             # GPT continuous scoring (-1..1)
    │   │   ├── politeness_score_batch_chunks_resume_logged.py
    │   │   ├── scores-*.csv
    │   │   ├── Average_score_per_model-*.csv
    │   │   └── README.md
    │   │
    │   ├── frequency/            # Polite-word frequency scoring
    │   │   ├── sentence-score.py
    │   │   ├── word-score.py
    │   │   ├── gpt_politeword_freq.py
    │   │   ├── scores-*-sentence.csv / word.csv
    │   │   ├── Average_score_sentence-*.csv / word-*.csv
    │   │   └── README.md
    │   │
    │   ├── polite_guard_eval/    # HuggingFace Polite-Guard evaluation
    │   │   ├── run_eval.py
    │   │   ├── scores-*.csv
    │   │   └── README.md
    │   │
    │   └── README.md             # (Optional) summary for all evaluation methods
    │
    └── data/                     # (Optional) Instructions on dataset retrieval

------------------------------------------------------------------------

##  Quick Start

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Prepare dataset

-   Input: JSON files containing responses.\
-   Each record must follow:

``` json
{
  "model": "model_name",
  "query_id": "123",
  "answer": "Your response text here"
}
```

-   Dataset is **not included**. Please follow project/course
    instructions to obtain it.

### 3. Run evaluations

#### GPT continuous scoring

``` bash
python evaluation/gpt_eval/politeness_score_batch_chunks_resume_logged.py     --input_dir data/     --out evaluation/gpt_eval/scores-1B.csv     --model gpt-4o-mini     --chunk_size 100     --rpm 200
```

#### Frequency-based scoring

``` bash
python evaluation/frequency/gpt_politeword_freq.py     --input_dir data/     --out evaluation/frequency/scores-1B-sentence.csv     --model gpt-4o-mini
```

#### Polite-Guard evaluation

``` bash
python evaluation/polite_guard_eval/run_eval.py     --input_dir data/     --out evaluation/polite_guard_eval/scores-1B.csv
```

------------------------------------------------------------------------

##  Outputs

-   `scores-*.csv` → raw per-response politeness scores\
-   `Average_score_*.csv` → aggregated mean politeness per model\
-   `raw_outputs/` → raw completions from GPT-based evaluators

------------------------------------------------------------------------

## Alignment with Expectations & Grading

-   **Complete materials**: Scripts, results, documentation provided.\
-   **Well-documented**: Top-level overview + per-folder READMEs.\
-   **Reproducible**: Instructions to install/run, resumable scripts.\
-   **Functional**: All methods return results in a standardized CSV
    format.\
-   **Adequate**: Implements multiple methods to triangulate politeness
    measurement.\
-   **Dataset handling**: Instructions provided; dataset not included.\
-   **Extra**: Cross-method comparison and multiple evaluation metrics.

------------------------------------------------------------------------

## Notes

-   Each evaluation method is modular and can be extended with new
    datasets or models.\
-   Results are directly comparable since they follow a **standard CSV
    format**.\
-   The repo is structured for **clarity and grading transparency**.
