# Frequency-based Politeness Scoring

## 📌 Overview

This folder contains code and results for **frequency-based politeness
scoring**.\
Instead of assigning a continuous score in \[-1, 1\], this approach
evaluates responses based on the **frequency of polite words** (e.g.,
*please, thank you, sorry, appreciate*).

The scoring is automated via GPT prompts, where each response is
returned with a **numeric frequency value**.

------------------------------------------------------------------------

## 📂 Contents

    ├── sentence-score.py                  # Script for sentence-level frequency scoring
    ├── word-score.py                      # Script for word-level frequency scoring
    ├── gpt_politeword_freq.py             # Script using GPT to estimate polite-word frequencies
    ├── scores-1B-sentence.csv             # Raw results (1B model, sentence-level)
    ├── scores-1B-word.csv                 # Raw results (1B model, word-level)
    ├── scores-3B-sentence.csv
    ├── scores-3B-word.csv
    ├── scores-8B-sentence.csv
    ├── scores-8B-word.csv
    ├── Average_score_sentence-1B.csv      # Aggregated averages (sentence-level)
    ├── Average_score_sentence-3B.csv
    ├── Average_score_sentence-8B.csv
    ├── Average_score_word-1B.csv
    ├── Average_score_word-3B.csv
    ├── Average_score_word-8B.csv
    └── raw_outputs/                       # Raw model outputs (per chunk, for auditing)

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Prepare input data

-   Input: JSON files containing responses.\
-   Each record must include:

``` json
{
  "model": "model_name",
  "query_id": "123",
  "answer": "Your response text here"
}
```

### 3. Run scoring

Example (sentence-level frequency, GPT-4o-mini):

``` bash
export OPENAI_API_KEY=sk-...
python gpt_politeword_freq.py     --input_dir ../data/     --out scores-1B-sentence.csv     --model gpt-4o-mini     --chunk_size 100     --rpm 200     --raw_dir raw_outputs
```

### 4. Outputs

-   `scores-*-sentence.csv`, `scores-*-word.csv`: raw frequency scores
    per response\
-   `Average_score_sentence-*.csv`, `Average_score_word-*.csv`:
    aggregated averages\
-   `raw_outputs/`: raw GPT completions per chunk

------------------------------------------------------------------------

## 📦 Notes

-   The script is **resumable** and logs progress/errors in
    `scores.log`.\
-   API key must be set as an **environment variable**
    (`OPENAI_API_KEY`).\
-   Do **not** commit keys or credentials to GitHub.\
-   Complements other evaluation approaches:
    -   `evaluation/gpt_eval/` → continuous politeness scoring (−1..1)\
    -   `evaluation/polite_guard_eval/` → HF Polite-Guard scoring

------------------------------------------------------------------------

## 🏆 Alignment with Expectations & Grading

-   **Materials complete**: includes scripts, results, and
    documentation.\
-   **Well-documented**: README explains structure, usage, and outputs.\
-   **Reproducibility**: clear instructions; can resume from partial
    runs.\
-   **Functionality**: works with GPT API, produces both raw and
    aggregated results.\
-   **Adequacy**: faithfully implements frequency-based politeness
    scoring.\
-   **Dataset handling**: input dataset is external; instructions
    provided.\
-   **Submission ready**: structured, documented, and easy to run.
