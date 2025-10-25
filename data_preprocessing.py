import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Dict
import json

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

Lemmatizer = WordNetLemmatizer()
StopWords = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase, remove punctuation and extra whitespace
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in StopWords and len(t) > 1]

def lemmatize(tokens: List[str]) -> List[str]:
    return [Lemmatizer.lemmatize(t) for t in tokens]

def pos_tag(tokens: List[str]):
    return nltk.pos_tag(tokens)

def preprocess_text(text: str) -> Dict:
    """Run full pipeline and return dictionary with intermediate results."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemm = lemmatize(filtered)
    pos = pos_tag(lemm)

    return {
    "original_text": text,
    "cleaned_text": cleaned,
    "tokens": tokens,
    "filtered_tokens": filtered,
    "lemmatized_tokens": lemm,
    "pos_tags": pos,
    }

def load_intents(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# quick test when run directly
if __name__ == '__main__':
    sample = "Hello! I'm Aya — how are you today?"
    print(preprocess_text(sample))