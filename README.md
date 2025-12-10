# Website :

- Technologies : JavaScript / HTML / CSS.
- Main Colors : #B02D3B, #732054
- Capabilities : Explain the challange's and allow the participants to submit their solutions as a zip file.
- Make at as pleasing to the eye as possible.

# Challange Description : 

## What is Black-Box attack ?

Black-box attack is a type of cybersecurity attack where the attacker — despite having no knowledge of the internal workings of a given system — uses trial and error and observes the system’s output for certain inputs to gain information about its internal behavior and reveal vulnerabilities, which can then be exploited.

Black-box attacks are even more concerning when it comes to machine learning systems that allow users to interact with a model through an interface such as an API endpoint, while the system does not want to make its model weights publicly available. A successful black-box attack can allow an unauthorized party to obtain a model with capabilities similar to the one being hosted.

## Toy Endpoint :

`POST /predict?k=K&threshold=THRESHOLD HTTP/1.1` allows users to interact with a model that is trained to predict keywords of **scientific articles** based on their abstracts. The endpoint allows bulk requests of maximum size of 64 document per request and returns the top-`K` most releveant keywords with a threshold higher than `THRESHOLD`. The endpoint expects the body to be in the following format : 

```json
{
    "documents" : [
        {
            "description" : "....",
        },
        {
            "description" : "....",
        }
    ]
}
```

And the response has the following format : 

```json
[
    [
        { "name" : "keyword_name", "rank" : 0 },
        { "name" : "keyword_name", "rank" : 1 },
        ...
    ],
    [
        { "name" : "keyword_name", "rank" : 0 },
        { "name" : "keyword_name", "rank" : 1 },
        ...
    ],
    ...
]
```

## Solution

**Submission :**

The solution should be sent as a file `[YOUR_TEAM_NAME].zip` that contains both inference code (should be named : `main.py`) and any other resources the inference code may need such as your model's weights. The inference code should not have access to the interent or to any file outside of its containing folder. view `inference.py` for a simple working submission example.

**Score :**

Your model similarity to the hosted model will be compared to the hosted model by comparing their predictions probabilities on the same hidden test-set more specifically the average soft jaccard index will be utilized : 

$$
\text{Soft Jaccard}_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} 
\frac{\sum_{k} \min(y_{ik}, \hat{y}_{ik})}{\sum_{k} \max(y_{ik}, \hat{y}_{ik})}
$$
