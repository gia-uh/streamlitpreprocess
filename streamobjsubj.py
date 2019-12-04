from read_info import load_file
import streamlit as st
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import spacy
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import preprocessing
from models import train_svm, plot_learning_curve, learning_curves_preprocess, test_cases
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import Counter
from joblib import dump, load
import os
# from models import train_svm

from bert_serving.client import BertClient

def to_obj_sub(value):
    if value == 'Objetivo':
        return value

    return 'Subjetivo'


def generate_dataframe(type, data):

    yudy_rule = st.checkbox("Correct match", key='correctMatch')

    # This is incorrect asumtion
    if yudy_rule:
        data = [value for value in data if max(Counter((v.answer for v in value.answers)).values())>=2 ]

    if 'Todo' == type:
        return pd.DataFrame([{'text': values.text, 'ans': values.answers[0].answer} for values in data])

    if 'Objetivo - Subjetivo' == type:
        return pd.DataFrame([{'text': values.text, 'ans': to_obj_sub(values.answers[0].answer)} for values in data])

    if "Positivo - Neutro - Negativo":
        return pd.DataFrame([{'text': values.text, 'ans': values.answers[0].answer} for values in data if values.answers[0].answer != 'Objetivo'])

    raise Exception(f"type is {type}")

file_path = st.text_input('Path', 'saved.json')

data = load_file(file_path)

types = ['Todo', "Positivo - Neutro - Negativo", 'Objetivo - Subjetivo']
t = st.selectbox('Tipo de dato', types, key='dataselector')

df = generate_dataframe(t,data)


df



d = df.describe()

d

gb = df.groupby('ans').count()
gb


labels = gb.index
values = gb.text

pieAns = go.Figure(data=[go.Pie(labels=labels, values=values)])

st.plotly_chart(pieAns)

le = preprocessing.LabelEncoder()
Y = le.fit_transform(df.ans)

st.markdown('****')

if st.checkbox('Count Vectorizer'):


    st.title('Models with count vectorizer')

    @st.cache(allow_output_mutation=True)
    def vectorizeTexts(texts):
        cv = CountVectorizer()
        X = cv.fit_transform(texts)
        X = X.toarray()
        return cv, X

    cv, X = vectorizeTexts(df.text)

    cv_model_name = st.text_input('Model Name', 'countvectorizer.joblib')
    dump(cv, os.path.join('models', cv_model_name))

    title = 'LinearSVC'

    test_cases(f'CountVectorizer {t}', X, Y, svm.LinearSVC(), le)



if st.checkbox('SpacyVect'):

    @st.cache(allow_output_mutation=True)
    def vectorizeTextsSpacy(texts):
        @st.cache(allow_output_mutation=True)
        def load_spacy():
            return spacy.load('es')

        nlp = load_spacy()
        X = []
        for t in texts:
            X.append(nlp(t).vector)
        return X
    X= vectorizeTextsSpacy(df.text)

    st.title('Models with Spacy tensor')

    test_cases(f'SpacyVectorizer {t}', X, Y, svm.LinearSVC(), le)



if st.checkbox('BertVect'):

    st.title('Models with count BERT')


    @st.cache(allow_output_mutation=True)
    def vectorizeBERT(texts):
        bc=BertClient(ip='10.6.122.217', timeout=30000)
        print(bc)
        resp = bc.encode(list(texts))
        return resp
    X= vectorizeBERT(df.text)



    test_cases(f'Bert {t}', X, Y, svm.LinearSVC(), le)

