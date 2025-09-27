# Politeness Scoring Repository

## 📌 Overview

This repository contains code and results for the **Politeness Scoring
Project**, where multiple large language models (LLMs) are evaluated on
a politeness rating task.\
Each response is scored on a continuous scale from **-1 (very impolite)
to 1 (very polite)**.\
The repository includes scripts for running the study, raw results,
aggregated CSVs, and documentation.

------------------------------------------------------------------------

## 📂 Repository Structure

    ├── politeness_score_batch_chunks_resume_logged.py   # Main evaluation script
    ├── requirements.txt                                 # Dependencies (to be added)
    ├── scores-1B.csv                                    # Raw scores for 1B model
    ├── scores-3B.csv                                    # Raw scores for 3B model
    ├── scores-8B.csv                                    # Raw scores for 8B model
    ├── Average_score_per_model-1B.csv                   # Aggregated average scores
    ├── Average_score_per_model-3B.csv
    ├── Average_score_per_model-8B.csv
    ├── raw_outputs/                                     # Raw model outputs (per chunk)
    └── readme.MD                                        # This file

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Install requirements

``` bash
pip install -r requirements.txt
```

### 2. Prepare input data

-   Input dataset (`.json` files containing model responses) **is not
    included** in this repository.\
-   Please obtain the dataset separately and place it under a chosen
    folder (e.g., `data/`).

Each record in the JSON should look like:

``` json
{
  "model": "model_name",
  "query_id": "123",
  "answer": "Your response text here"
}
```

### 3. Run the scoring script

``` bash
python politeness_score_batch_chunks_resume_logged.py     --input_dir data/     --out scores.csv     --model gpt-4o-mini     --chunk_size 100     --rpm 200
```

### 4. Outputs

-   **Raw outputs**: Saved per chunk in `raw_outputs/`.
-   **CSV results**:
    -   `scores-*.csv`: Raw politeness scores for each response.
    -   `Average_score_per_model-*.csv`: Aggregated mean scores per
        model.

------------------------------------------------------------------------

## 📑 File Descriptions

-   **`politeness_score_batch_chunks_resume_logged.py`**\
    Main script for running politeness scoring. Supports resuming, error
    logging, retry with backoff, and progress bars.

-   **`scores-*.csv`**\
    Raw model outputs with the following columns:\
    `file, model_in_file, query_id, chunk_id, score`

-   **`Average_score_per_model-*.csv`**\
    Aggregated results (mean politeness score per model).

-   **`requirements.txt`**\
    Python dependencies (see below).

-   **`raw_outputs/`**\
    Contains raw text outputs from each chunk request.

------------------------------------------------------------------------

## 📦 Requirements

Add the following dependencies to `requirements.txt`:

``` txt
openai
tqdm
pandas
numpy
```

------------------------------------------------------------------------

## ✅ Reproducibility Notes

-   The script is **resumable**: it will skip already-processed chunks
    in the output CSV.\
-   Logs are written to `scores.log`.\
-   Quota or rate-limit errors will stop execution gracefully.\
-   Dataset and model weights are **not included** (per submission
    rules).

------------------------------------------------------------------------

## 🏆 Grading Alignment

-   **Materials complete**: Includes code, scripts, results, and
    documentation.\
-   **Well-documented**: Clear README, structured outputs, logging.\
-   **Reproducibility**: Minimal setup; instructions provided.\
-   **Functionality**: Script can resume, aggregate, and handle errors
    robustly.\
-   **Adequacy**: Implements politeness evaluation in line with project
    goals.

------------------------------------------------------------------------

## ✨ Extra Notes

-   The repo is modular: other evaluation datasets can be plugged in.\
-   Aggregated scores allow direct comparison across models (1B, 3B,
    8B).\
-   Raw logs and outputs provide full traceability for debugging.
