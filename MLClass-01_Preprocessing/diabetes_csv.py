#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.
Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.
@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

import requests



print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
data_app = pd.read_csv('diabetes_app.csv')


# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Glucose', 'BMI', 'Age']
X = data[feature_cols]
y = data.Outcome

# Separa os dados entre treinamento e teste utilizando validação cruzada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.03, random_state = 5)

# Completando os valores que faltam com o valor mais frequente entre eles
imp = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

# Aplica o fit transform no X_train
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

# Escala cada atributo pelo seu valor absoluto
scalar = MaxAbsScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=11)

# Treina com as bases de treinamento
neigh.fit(X_train, y_train)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = data_app[feature_cols]
data_app = scalar.transform(data_app)
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "http://aydanomachado.com/mlclass/01_Preprocessing.php"

# Chave id da equipe
DEV_KEY = "AV"

# Json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")