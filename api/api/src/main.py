import constants as C
from fastapi import FastAPI, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.ml import load_model, Recommender
from src.model import Category, DocumentsList
from typing import List, Dict

### Create App
app = FastAPI()

### Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Load Model
load_model(C.MODEL_PATH)

### PREDICTION ENDPOINT
@app.post("/predict")
def predict(
    model : Recommender = Depends(lambda: load_model(C.MODEL_PATH)),
    body : DocumentsList = Body(),
    threshold : float = Query(default=0.0, ge=0.0, le=1.0),
    k : int = Query(default=5, ge=1, le=10),
) -> List[List[Category]]:
    
    venues = model.get_venues(
        articles=list(map(lambda doc : doc.description,body.documents)),
        batch_size=C.BATCH_SIZE,
        threshold=threshold,
        top_k=k
    )

    return [
        [ 
            Category(name=keyword[0], rank=i)
            for i,keyword in enumerate(document_venues)
        ]
        for document_venues in venues
    ]

### METADATA ENDPOINT
@app.get("/metadata")
def metadata() -> Dict:

    return {
        "document_min_tokens" : C.MIN_TOKENS,
        "document_max_tokens" : C.MAX_TOKENS,
        "tokenizer" : "ntlk.word_tokenize",
        "max_number_of_docs_per_request" : C.MAX_DOCUMENTS,
    }

### METADATA ENDPOINT
@app.get("/labels")
def metadata(
    model : Recommender = Depends(lambda: load_model(C.MODEL_PATH))
) -> list[str]:
    return model.venues