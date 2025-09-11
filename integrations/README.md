
<h1 align="center"> PolitenessLayers integration</h1>



---
##  Structure
```
my_api_project/
├── app.py                 # FastAPI entrypoint (auto-wrapping)
├── myapi/
│   ├── __init__.py
│   └── api.py            # Code extracted from your notebook (comments are in English)
├── requirements.txt       # Includes FastAPI + PolitenessLayers (from GitHub)
├── .gitignore
└── README.md
```

Politeness scoring with OpenAI models (gpt-4o-mini by default).

Caching + backoff to handle API rate limits (429).

PromptConfig class for controlling:

scoring scale (default [-1, 1])

instruction strictness

output format rules

few-shot examples (with shuffling option)

Batch evaluation with adjustable batch size and RPM (requests per minute).

Evaluation metrics: Spearman, Pearson, MAE for regression; Accuracy/F1 for classification.

Data integration with Convokit politeness corpora
:

Wikipedia politeness

StackExchange politeness

Stratified sampling: sample utterances evenly across politeness bins.
