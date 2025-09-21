
# Politeness Scoring
This folder contains data and scripts for evaluating **politeness scores (word & sentence level)** of model outputs after pruning.

# Reference
- Polite words dictionary (feature_table): https://myeomans.r-universe.dev/politeness

## Files

- **word-score.py**  
  Main Python script for batch polite word frequency scoring.  
  - Reads multiple JSON files with model outputs.  
  - Sends them to OpenAI (`gpt-4o-mini`) for politeness scoring.  
  - Saves results in CSV with resume support and logging.
  
- **sentence-score.py**  
  Main Python script for batch polite sentence frequency scoring.  
  - Reads multiple JSON files with model outputs.  
  - Sends them to OpenAI (`gpt-4o-mini`) for politeness scoring.  
  - Saves results in CSV with resume support and logging.

- **scores-1B-word.csv**  
  Raw polite word frequency scores (number of polite words / number of sentences) for all responses from the **LLaMA3.2-1B** pruning experiment.

- **scores-3B-word.csv**  
  Raw polite word frequency scores (number of polite words / number of sentences) for all responses from the **LLaMA3.2-3B** pruning experiment.

- **scores-8B-word.csv**  
  Raw polite word frequency scores (number of polite words / number of sentences) for all responses from the **LLaMA3.1-8B** pruning experiment.

- **scores-1B-sentence.csv**  
  Raw polite sentence frequency scores (number of polite sentences / number of sentences) for all responses from the **LLaMA3.2-1B** pruning experiment.

- **scores-3B-sentence.csv**  
  Raw polite sentence frequency scores (number of polite sentences / number of sentences) for all responses from the **LLaMA3.2-3B** pruning experiment.

- **scores-8B-sentence.csv**  
  Raw polite sentence frequency scores (number of polite sentences / number of sentences) for all responses from the **LLaMA3.1-8B** pruning experiment.
  
- **Average_score_word-1B.csv**  
  Aggregated results: average polite word frequency score for each pruned layer of the **1B** model.

- **Average_score_word-3B.csv**  
  Aggregated results: average polite word frequency score for each pruned layer of the **3B** model.
  
- **Average_score_word-8B.csv**  
  Aggregated results: average polite word frequency score for each pruned layer of the **8B** model.

- **Average_score_sentence-1B.csv**  
  Aggregated results: average polite sentence frequency score for each pruned layer of the **1B** model.

- **Average_score_sentence-3B.csv**  
  Aggregated results: average polite sentence frequency score for each pruned layer of the **3B** model.
  
- **Average_score_sentence-8B.csv**  
  Aggregated results: average polite sentence frequency score for each pruned layer of the **8B** model.
