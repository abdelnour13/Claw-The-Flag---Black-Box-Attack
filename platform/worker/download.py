import nltk
import spacy
import spacy.cli
from textblob import download_corpora

# -----------------------
# NLTK essentials
# -----------------------
nltk_packages = [
    "punkt",        # tokenizers
    "wordnet",      # WordNet lexical database
    "stopwords",    # stop words
    "omw-1.4",      # WordNet multilingual
    "averaged_perceptron_tagger",  # POS tagging
    "vader_lexicon", # sentiment analysis
    "maxent_ne_chunker", # NER
    "words"         # English word list
]

for pkg in nltk_packages:
    try:
        nltk.download(pkg)
        print(f"NLTK package downloaded: {pkg}")
    except Exception as e:
        print(f"Failed to download NLTK package {pkg}: {e}")

# -----------------------
# spaCy essentials
# -----------------------
# Common English models
spacy_models = [
    "en_core_web_sm",  # small English model
    "en_core_web_md",  # medium English model
    "en_core_web_lg",  # large English model
]

for model in spacy_models:
    try:
        spacy.cli.download(model)
        print(f"spaCy model downloaded: {model}")
    except Exception as e:
        print(f"Failed to download spaCy model {model}: {e}")

# -----------------------
# TextBlob essentials
# -----------------------
try:
    download_corpora.download_all()
    print("TextBlob corpora downloaded")
except Exception as e:
    print(f"Failed to download TextBlob corpora: {e}")

print("All NLP resources downloaded successfully!")
