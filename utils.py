import random
import joblib
from typing import List

def load_models(models_dir: str = './models'):
    vec = joblib.load(f"{models_dir}/count_vectorizer.joblib")
    nb = joblib.load(f"{models_dir}/naive_bayes.joblib")
    dt = joblib.load(f"{models_dir}/decision_tree.joblib")
    return vec, nb, dt

def choose_response(intent_tag: str, intents: dict) -> str:
    for it in intents.get('intents', []):
        if it['tag'] == intent_tag:
            return random.choice(it.get('responses', ["Sorry, I don't have an answer for that."]))
    return "Sorry, I don't have an answer for that."