
# Prompt Debugger – Politeness Scoring

## ✨ Features
- **Politeness scoring** with OpenAI models (`gpt-4o-mini` by default).  
- **Caching + backoff** to handle API rate limits (429).  
- **PromptConfig class** for controlling:  
  - scoring scale (default `[-1, 1]`)  
  - instruction strictness  
  - output format rules  
  - few-shot examples (with shuffling option)  
- **Batch evaluation** with adjustable batch size and RPM (requests per minute).  
- **Evaluation metrics**: Spearman, Pearson, MAE for regression; Accuracy/F1 for classification.  
- **Data integration** with Convokit politeness corpora:  
  - Wikipedia politeness  
  - StackExchange politeness  
- **Stratified sampling**: sample utterances evenly across politeness bins.  

---

## 📝 Final Prompt

```text
You are a politeness judge. Rate on -1..1. Reply with ONLY a number. 
If you output anything else, you fail. Return a number.

Output only a number in [-1, 1]. No words, no explanation, no units.

Examples:
Response: "Can you look at this when you have a chance?"
Politeness score: 0.7
Response: "I would appreciate it if you could clarify this point."
Politeness score: 0.8
Response: "Please fix this error."
Politeness score: 0.0
Response: "Why didn't you follow the instructions properly?"
Politeness score: -0.5
Response: "Shut up, you clearly don't understand this."
Politeness score: -1.0

Now evaluate the following new response:
Response: "<sentence>"
Politeness score:
```
