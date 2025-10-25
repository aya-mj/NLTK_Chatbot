import streamlit as st
from streamlit_chat import message
import joblib
import json

from data_preprocessing import preprocess_text
from utils import load_models, choose_response

st.set_page_config(page_title='NLTK Chatbot', layout='centered')


st.title('NLTK Chatbot')
st.write('A simple chatbot demo using NLTK + scikit-learn (Bag-of-Words).')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

vec, nb_model, dt_model = load_models('./models')


# Sidebar
model_choice = st.sidebar.selectbox('Choose model', ['Naive Bayes', 'Decision Tree'])
show_preproc = st.sidebar.checkbox('Show preprocessing', value=True)
show_probs = st.sidebar.checkbox('Show prediction probabilities', value=True)


# Session state for chat
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input('You: ', '')

if st.button('Send') and user_input.strip() != '':
    # preprocessing
    p = preprocess_text(user_input)
    cleaned_joined = ' '.join(p['lemmatized_tokens'])
    vec_in = vec.transform([cleaned_joined])

    if model_choice == 'Naive Bayes':
        pred = nb_model.predict(vec_in)[0]
        probs = nb_model.predict_proba(vec_in)[0]
        classes = nb_model.classes_
    else:
        pred = dt_model.predict(vec_in)[0]
        # DecisionTreeClassifier may not implement predict_proba for some settings but usually does
        try:
            probs = dt_model.predict_proba(vec_in)[0]
            classes = dt_model.classes_
        except Exception:
            probs = None
            classes = dt_model.classes_
    response = choose_response(pred, intents)

    st.session_state.history.append({'You': user_input, 'chatbot': response, 'preproc': p, 'pred': pred, 'probs': (classes, probs)})

for i, turn in enumerate(st.session_state.history):
    message(turn['You'], is_user=True, key=f"user_{i}")
    message(turn['chatbot'], key=f"bot_{i}")

if st.session_state.history and show_preproc:
    last = st.session_state.history[-1]
    st.markdown("**Preprocessing Details:**")
    st.write(last['preproc'])

if st.session_state.history and show_probs:
    last = st.session_state.history[-1]
    classes, probs = last.get('probs', (None, None))
    if probs is not None:
        st.markdown("**Prediction Probabilities:**")
        prob_dict = {cls: float(f"{prob:.4f}") for cls, prob in zip(classes, probs)}
        st.write(prob_dict)
    else:
        st.write("Prediction probabilities not available for the selected model.")