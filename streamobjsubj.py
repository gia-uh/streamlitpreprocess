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
# from models import train_svm

from bert_serving.client import BertClient

def to_obj_sub(value):
    if value == 'Objetivo':
        return value

    return 'Subjetivo'

file_path = st.text_input('Path', 'saved.json')

data = load_file(file_path)

df = pd.DataFrame([{'text': values.text, 'ans': to_obj_sub(values.answers[0].answer)} for values in data])


df

btn_todo = st.button("todo")
btn_pnn = st.button("Positivo - Neutro - Negativo")
btn_os = st.button("Objetivo - Subjetivo")



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


    title = 'LinearSVC'

    test_cases('CountVectorizer', X, Y, svm.LinearSVC(), le)



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

    test_cases('SpacyVectorizer', X, Y, svm.LinearSVC(), le)



if st.checkbox('BertVect'):

    st.title('Models with count BERT')


    @st.cache(allow_output_mutation=True)
    def vectorizeBERT(texts):
        bc=BertClient(ip='10.6.122.217', timeout=30000)
        print(bc)
        resp = bc.encode(list(texts))
        return resp
    X= vectorizeBERT(df.text)



    test_cases('Bert', X, Y, svm.LinearSVC(), le)

