# Polite-Guard Evaluation (HuggingFace)

##  Overview

This module evaluates politeness using **HuggingFace's Polite-Guard
model**\
(https://huggingface.co/Intel/polite-guard).\
It provides **classification-based politeness judgments** and supports
**layer-wise analysis**.

------------------------------------------------------------------------

##  Contents

    polite_guard_eval/
    ├── run_eval.py                 # Main script for running Polite-Guard evaluation
    ├── scores-layer1.csv           # Example: layer-wise results
    ├── scores-layer2.csv
    ├── scores-final.csv            # Final aggregated results
    └── README.md

------------------------------------------------------------------------

## How to Run

``` bash
python run_eval.py     --input_dir ../data/     --out scores-final.csv     --layers all
```

Options: - `--layers`: choose which hidden layers to extract (e.g.,
`1,5,10` or `all`)\
- `--out`: CSV file to save results

------------------------------------------------------------------------

##  Outputs

-   `scores-layer*.csv`: Layer-wise politeness scores per response\
-   `scores-final.csv`: Final aggregated politeness scores

------------------------------------------------------------------------

##  Notes

-   This evaluation **does not require an API key**, only HuggingFace
    models.\
-   Results are comparable to GPT-based and frequency-based scoring.\
-   Useful for **probing intermediate representations** of politeness.
