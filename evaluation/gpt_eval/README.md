# GPT-based Continuous Politeness Scoring

##  Overview

This module evaluates responses using **GPT models** as judges.\
Each response is assigned a **continuous politeness score in \[-1,
1\]**:

-   `-1`: very impolite\
-   `0`: neutral / directive\
-   `+1`: very polite

This captures **subtle tone differences** that frequency-based methods
may miss.

------------------------------------------------------------------------

##  Contents

    gpt_eval/
    ├── politeness_score_batch_chunks_resume_logged.py   # Main scoring script
    ├── scores-1B.csv                                    # Raw scores (1B model)
    ├── scores-3B.csv                                    # Raw scores (3B model)
    ├── scores-8B.csv                                    # Raw scores (8B model)
    ├── Average_score_per_model-1B.csv                   # Aggregated averages
    ├── Average_score_per_model-3B.csv
    ├── Average_score_per_model-8B.csv
    ├── raw_outputs/                                     # Raw model completions
    └── README.md

------------------------------------------------------------------------

##  How to Run

``` bash
export OPENAI_API_KEY=sk-...
python politeness_score_batch_chunks_resume_logged.py     --input_dir ../data/     --out scores-1B.csv     --model gpt-4o-mini     --chunk_size 100     --rpm 200     --raw_dir raw_outputs
```

------------------------------------------------------------------------

##  Outputs

-   `scores-*.csv`: Raw scores per response
    (`file, model_in_file, query_id, chunk_id, score`)\
-   `Average_score_per_model-*.csv`: Aggregated mean politeness scores\
-   `raw_outputs/`: Raw GPT completions (for auditing)

------------------------------------------------------------------------

##  Notes

-   Script is resumable: already-completed chunks are skipped.\
-   Error handling: retry with exponential backoff, logs stored in
    `scores.log`.\
-   Dataset is **not included** (instructions in top-level README).
