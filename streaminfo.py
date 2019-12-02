from read_info import load_file
import streamlit as st
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
from minio import Minio
import json
import numpy as np

minio_config = st.text_input('MinioInfoData', 'miniodata.json')
file_path = st.text_input('Path', 'saved.json')


if st.button('Update Data from server'):

    config = json.load(open(minio_config))
    minioClient = Minio(config['host'],
                        access_key=config['access_key'],
                        secret_key=config['secret_key'],
                        secure=False)
    response = minioClient.get_object('metaclassifier', 'data/saved.json')

    with open(file_path, 'wb') as file:
        file.write(response.data)


data = load_file(file_path)

f"""
Se han clasificado un total de {len(data)} preguntas
"""

ans = []

for q in data:
    ans.extend(q.answers)

f"""
un total de {len(ans)} clasificaciones

(pueden existir repetidas si se clasifica una noticia más de una vez)
"""

users = {}
count = Counter()

for user in ans:
    users[user.extra_data['user']['id']] = user.extra_data['user']
    count[user.extra_data['user']['id']] += 1
    users[user.extra_data['user']['id']
          ]['classified'] = count[user.extra_data['user']['id']]

df = pd.DataFrame([values for values in users.values()])

df

labels = [f'{name} ({username})' for name,
          username in zip(df.first_name, df.username)]
values = df.classified

pieUsers = go.Figure(data=[go.Pie(labels=labels, values=values)])
st.plotly_chart(pieUsers)




df = pd.DataFrame([dict(**{'text': values.text},
    **Counter(map(lambda x: x.answer, values.answers))) for values in data])
df = df.fillna(0)
df

ds = df.describe()
ds


df = df[np.max([df.Positivo, df.Negativo, df.Neutro, df.Objetivo], axis=0) >= 2]

df

ds = df.describe()

ds


# Bug: voy a dividir entre 2, esta no es la expresión correcta
labels = ['Positivo', 'Negativo', 'Neutro', 'Objetivo']
values = np.array([df.Positivo.sum(), df.Negativo.sum(), df.Neutro.sum(), df.Objetivo.sum()])/2

pieAns = go.Figure(data=[go.Pie(labels=labels, values=values)])

st.plotly_chart(pieAns)
