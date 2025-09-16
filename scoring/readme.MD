
# Politeness Scoring Experiments

This folder contains data and scripts for evaluating **politeness scores** of model outputs after pruning.

## Files

- **politeness_score_batch_chunks_resume_logged.py**  
  Main Python script for batch scoring.  
  - Reads multiple JSON files with model outputs.  
  - Sends them to OpenAI (`gpt-4o-mini`) for politeness scoring.  
  - Saves results in CSV with resume support and logging.

- **scores-1B.csv**  
  Raw politeness scores for all responses from the **LLaMA3.2-1B** pruning experiment.

- **scores-8B.csv**  
  Raw politeness scores for all responses from the **LLaMA3.1-8B** pruning experiment.

- **Average_score_per_model-1B.csv**  
  Aggregated results: average politeness score for each pruned layer of the **1B** model.

- **Average_score_per_model-8B.csv**  
  Aggregated results: average politeness score for each pruned layer of the **8B** model.
