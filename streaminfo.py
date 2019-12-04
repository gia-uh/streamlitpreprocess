from read_info import load_file
import streamlit as st
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
from minio import Minio
import json
import numpy as np
from datetime import datetime
import os

minio_config = st.text_input('MinioInfoData', 'miniodata.json')
file_path = st.text_input('Path', 'saved.json')

def merge(data, datam):
    datad = {i['id']:i for i in data}
    datam = {i['id']:i for i in datam}
    for idd, datm in datam.items():
        if idd in datad:
            datd = datad[idd]
            ansm = {i['extra_data']['user']['id']:i for i in datm['answers']}
            ansd = {i['extra_data']['user']['id']:i for i in datd['answers']}
            for ida,ans in ansm.items():
                if ida in ansd:
                    t1 = ansd[ida]['extra_data']
                    t1 = t1.get('time','"2000-01-01 00:01:00.0000"')
                    t1 = datetime.strptime(t1.split('.')[0],'%Y-%d-%m %H:%M:%S')
                    t2 = ansm[ida]['extra_data']
                    t2 = t2.get('time','"2000-01-01 00:01:00.0000"')
                    t2 = datetime.strptime(t2.split('.')[0],'%Y-%d-%m %H:%M:%S')
                    if t1<t2:
                        ansd[ida]=ans
                else:
                    ansd[ida]=ans
            datad[idd]['answers'] = [i for i in ansd.values()]
        else:
            datad[idd]=datm
    return [i for i in datad.values()]


if st.button('Update Data from server'):

    config = json.load(open(minio_config))
    minioClient = Minio(config['host'],
                        access_key=config['access_key'],
                        secret_key=config['secret_key'],
                        secure=False)
    response = minioClient.get_object('metaclassifier', 'data/saved.json')
    datam = json.loads(response.data)
    if os.path.exists(file_path):
        data = json.load(open(file_path))
    else:
        data = {}

    with open(file_path, 'w') as file:
        merged = merge(data, datam)
        file.write(json.dumps(merged, indent=2))


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
