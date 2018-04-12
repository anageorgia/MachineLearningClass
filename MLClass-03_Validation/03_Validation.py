import pandas
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pandas.read_csv('abalone_dataset.csv')
data_app = pandas.read_csv('abalone_app.csv')

print('\n - Tipos dos dados:')
print(data.dtypes)

sex_col_numerical = {"sex": {"M": 0, "F": 1, "I": 2}}

data.replace(sex_col_numerical, inplace=True)
print('\n\n - Data head:\n')
print(data.head())

data_app.replace(sex_col_numerical, inplace=True)
print('\n - Data_app head:\n')
print(data_app.head())

print('\n\n - Criando X e y para o algoritmo de aprendizagem a partir do arquivo abalone_dataset')

feature_cols = ['sex','length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight','shell_weight']
X = data[feature_cols]
y = data.type

# Separa os dados entre treinamento e teste utilizando validação cruzada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# Criando o modelo preditivo para a base trabalhada
# print(' - Criando modelo preditivo Decision Tree Classifier:')

# ----------------------
# # training a DescisionTreeClassifier
#
# dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
# dtree_predictions = dtree_model.predict(X_test)
#
# # creating a confusion matrix
# cm = confusion_matrix(y_test, dtree_predictions)
#
# print(cm)
#
# accuracy_dtree = accuracy_score(y_test, dtree_predictions)
#
# print(accuracy_dtree)

# ----------------------
# training a linear SVM classifier
print('\n\n ->>>  |Criando modelo preditivo SVM|  <<<-\n')
from sklearn.svm import SVC

#svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_model_linear = SVC(C=3.4, kernel='linear', tol=0.01).fit(X_train, y_train)
# Description:
# C : float, optional (default=1.0) - Penalty parameter C of the error term.
# kernel: string, optional (default=’rbf’) - Specifies the kernel type to be used in the algorithm. (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable)
# tol: float, optional (default=1e-3) - Tolerance for stopping criterion.

svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

print('\n - Matriz de confusão:\n')
print(cm)

accuracy_svm = accuracy_score(y_test, svm_predictions)

print('\n\n - Acurácia SVM:\n')
print(accuracy_svm)

# -----------------------
# print(' - Criando modelo preditivo KNN:')
# # training a KNN classifier
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
#
# # accuracy on X_test
# accuracy = knn.score(X_test, y_test)
#
# # creating a confusion matrix
# knn_predictions = knn.predict(X_test)
# cm = confusion_matrix(y_test, knn_predictions)
#
# print(cm)
# print(accuracy)

# -----------------------
# print(' - Criando modelo preditivo Naive Bayes:')
# # training a Naive Bayes classifier
# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB().fit(X_train, y_train)
# gnb_predictions = gnb.predict(X_test)
#
# # accuracy on X_test
# accuracy = gnb.score(X_test, y_test)
#
# # creating a confusion matrix
# cm = confusion_matrix(y_test, gnb_predictions)
#
# print(cm)
# print (accuracy)

# -----------------------
# realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')

data_app = data_app[feature_cols]

y_pred = svm_model_linear.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

DEV_KEY = "AV"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
         'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
