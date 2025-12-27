# Black-Box Attack Challenge

## Challenge Details

### What is a Black-Box Attack?

A **black-box attack** is a type of cybersecurity attack where the attacker has **no knowledge of the internal workings** of a system. Instead, they interact with it through inputs and observe outputs to infer behavior and uncover vulnerabilities.

This is particularly concerning for **machine learning systems** exposed via APIs. Even without access to model weights, repeated querying can allow an attacker to reconstruct a model with capabilities similar to the hosted one.

---

## Toy Endpoint

```
POST /predict?k=K&threshold=THRESHOLD
```

This endpoint allows interaction with a model trained to predict **venues of scientific articles** based on their abstracts.

**Key properties:**

* Maximum batch size: **64 documents per request**
* Returns **Top-K** most relevant venues
* Results filtered by a relevance threshold
* Predictions are ranked by relevance
* Each document must contain **16–300 tokens** (tokenized using `nltk.word_tokenize`)

---

## Request Format

```json
{
  "documents": [
    { "description": "..." },
    { "description": "..." }
  ]
}
```

---

## Response Format

```json
[
  [
    { "name": "keyword_name", "rank": 0 },
    { "name": "keyword_name", "rank": 1 }
  ],
  [
    { "name": "keyword_name", "rank": 0 },
    { "name": "keyword_name", "rank": 1 }
  ]
]
```

---

## Helper Endpoint

```
GET /labels
```

Use this endpoint to retrieve the index of each label. This helps ensure your model’s predictions align with the hosted model’s label ordering.

---

## Scoring Method

Your model will be evaluated by comparing its prediction probabilities against the hosted model on a hidden test set using the **average soft Jaccard index**:

```
Soft Jaccard_avg = (1/N) * Σ_i [ Σ_k min(y_ik, ŷ_ik) / Σ_k max(y_ik, ŷ_ik) ]
```

Where:

* `y` = hosted model predictions
* `ŷ` = your model predictions

If your similarity score exceeds **`0.75`**, you will receive the flag confirming your winner status.

---

## What Do We Know?

* The hosted model uses **SentenceTransformer("all-MiniLM-L6-v2")** embeddings
* No specific preprocessing is applied
* The training dataset is unknown

---

## Submission Requirements

Submissions must be provided as a **`.zip` file** containing inference code that runs on a hidden test set.

At runtime, your code will have access to:

* `embeddings.npy`: Sentence-BERT embeddings of the hidden test articles

Your task is to estimate the probability of each **article–venue** pair.

**Requirements:**

* File name: **`[YOUR_TEAM_NAME].zip`**
* Maximum size: **`10M`**
* Must contain `main.py`
* Include all model weights and required resources
* Must generate a `scores.npy` file
* See the provided working example for reference

---

## Environment Constraints

* Maximum execution time: **`90 seconds`**
* Maximum memory usage: **`8G`**
* GPU availability: **`FALSE`**

---

## Advice

To ensure your submission runs correctly:

* Use device-agnostic code:

  ```python
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  ```
* Load checkpoints on the correct device:

  ```python
  torch.load(..., map_location=DEVICE)
  ```
* Test inference runtime locally first (test set contains ~7k articles) to avoid timeouts
