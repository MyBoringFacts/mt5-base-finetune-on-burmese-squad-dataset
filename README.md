# Burmese Question Answering with mT5  
Fine-tuning a Generative QA Model for Low-Resource Burmese  
**Internship Project at Language Understanding Laboratory (LLU), Thailand**

---

## Overview

This project aims to develop a **Burmese Question Answering (QA)** system by fine-tuning the **mT5** multilingual transformer model using a **generative approach**.

Traditional extractive QA models are not ideal for Burmese due to tokenization inconsistencies. Instead, this project uses **generative QA**, where the model receives a question and context and generates the answer directly.

This work was completed during my internship at the **Language Understanding Laboratory (LLU), Thailand**.

---

## Project Steps

### 1. Dataset Translation
- Source: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
- Translated from English to **Burmese** using:
  - [Meta’s NLLB (No Language Left Behind)](https://ai.facebook.com/research/no-language-left-behind/)
  - Google Gemini  API

### 2.  Exploratory Data Analysis (EDA)
- Dataset was large → performed sampling to create a manageable subset.
- Focused on maintaining balance and answer presence across examples.

### 3. Data Formatting
- Reformatted each sample into: question: <Burmese question> context: <Burmese context>
- Tokenization strategy:
  -  Truncation applied only to the context (`truncation="only_second"`)
  -  Used **stride** to create overlapping context chunks → ensures answers aren’t cut off.

---

## Model & Training

### Model
- Base: `mT5-base` (`AutoModelForSeq2SeqLM`)
- Tokenizer: `AutoTokenizer` from Hugging Face

###  Fine-Tuning Setup
- Hugging Face `Trainer` used with:
- `predict_with_generate=True` → model generates text during evaluation
- `load_best_model_at_end=True`
- Tracked metric: `f1` (or `exact_match`)
- Standard hyperparameters:
- Batch size, learning rate, 5 epochs, etc.

---

## Evaluation Metrics

Evaluation is done after each epoch using:
- **Exact Match (EM)**: % of answers exactly matching the gold answer (after normalization)
- **F1 Score**: Token-level overlap between predicted and gold answers

---

## Inference

After training, the fine-tuned model can be used for inference:

```python
inputs = tokenizer("question: ... context: ...", return_tensors="pt")
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```


## Acknowledgements

This project was carried out during my internship at the
Language Understanding Laboratory (LLU), Thailand.
Thanks to the mentors and researchers for guidance in low-resource NLP research.
