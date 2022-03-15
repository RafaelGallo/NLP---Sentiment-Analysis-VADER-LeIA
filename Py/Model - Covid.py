#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Versão do python

from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[2]:


# Importação das bibliotecas 

import pandas as pd # Pandas carregamento csv
import numpy as np # Numpy para carregamento cálculos em arrays multidimensionais

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Configuração para os gráficos largura e layout dos graficos

plt.style.use('fivethirtyeight')


# In[4]:


# NLTK para NLP
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download("stopwords")
nltk.download('punkt')


# In[5]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# # Base dados

# In[6]:


df = pd.read_csv("Corona_NLP_train.csv")
df


# In[7]:


# Exibido 5 primeiros dados
df.head()


# In[8]:


# Exibido 5 últimos dados 
df.tail(5)


# In[9]:


# Número de linhas e colunas
df.shape


# In[10]:


# Verificando informações das variaveis
df.info()


# In[11]:


# Tipo dos dados
df.dtypes


# In[12]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(df.shape[0]))
print("Números de colunas: {}" .format(df.shape[1]))


# In[13]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", df.isnull().sum().values.sum())


# In[14]:


# Verificando o total de sentimentos
df["Sentiment"].value_counts()


# In[15]:


# Verificando base
df.value_counts()


# # Análise de dados

# In[16]:


plt.figure(figsize=(20.5, 15))

plt.title("Análise sentimento das frases")
ax = sns.countplot(df["Sentiment"])
plt.xlabel("Sentimento")
plt.ylabel("Total")


# In[17]:


# Nuvem de palavras
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

todos_palavras = ' '.join([texto for texto in df["OriginalTweet"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# # Pré - processamento

# In[18]:


# Contagem de texto
df.OriginalTweet.count()


# In[19]:


# Dados duplicados

df.drop_duplicates(["OriginalTweet"], inplace = True)
df.OriginalTweet.count()


# # Treino e Teste
# 
# - Treino e teste da base de dados da coluna Review, sentimento

# In[26]:


# Variável para teste
treino = df["OriginalTweet"]

# Variável para treino
teste = df["Sentiment"]


# In[27]:


# Total de linhas e colunas dados variável x
treino.shape


# In[28]:


# Total de linhas e colunas dados variável y
teste.shape


# In[29]:


# Dados de limpeza para modelo PLN

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Removendo stop words
def remove_stop_words(instancia):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

# stemming no dataset
def text_stemming(instancia):
    stemmer = nltk.stem.RSLPStemmer()

    palavras = []

    for w in instancia.split():
        palavras.append(stemmer.stem(w))
    return (" ".join(palavras))

# Limpando base de dados
def dados_limpos(instancia):
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instancia)

# Redução de palavras flexionadas
def Lemmatization(instancia):
    
    palavras = []
    
    for w in instancia.split():
        palavras.append(wordnet_lemmatizer.lemmatize(w))
        return (" ".join(palavras))

# Pré-processamento removendo stopword e removendo caracteres indesejados.
def Preprocessing(instancia):
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','').replace('"','')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))


# In[30]:


# Visualizando texto
treino = [Preprocessing(i) for i in treino]
treino[:20]


# # Escalonamento dos dados

# In[31]:


from sklearn.preprocessing import LabelEncoder

# Nome do encoder
label = LabelEncoder()

# Transformação do dados
df["Sentiment"] = label.fit_transform(df["Sentiment"])

# Visualizção dos dados
df.head()


# In[32]:


# Word tokenize - E o processo de devidir uma string, textos e uma lista tokens 
# Modelo criado para classificar tweets positivos, negativos
# CountVectorizer criar um vocabulário de palavras e retornar em vetor

# Importação da biblioteca
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

# Nome token
tokenizer = TweetTokenizer()

# treinamento do vectorizer
vectorizer = CountVectorizer(analyzer ="word", 
                             tokenizer = tokenizer.tokenize)

# Visualizando os dados transformados para vetor
freq = vectorizer.fit_transform(treino)
freq.shape


# # Modelo Machine learning
# 
# - Modelo 01 - Naive bayes

# In[33]:


# Modelo machine learning - 1

# Importação da biblioteca sklearn o modelo Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Nome do algoritmo M.L
model_naive_bayes = MultinomialNB()

# Treinamento do modelo
model_naive_bayes_fit = model_naive_bayes.fit(freq, teste)

# Score do modelo
model_naive_bayes_score = model_naive_bayes.score(freq, teste)

print("Score - Modelo Naive bayes multinomialNB: %.2f" % (model_naive_bayes_score * 100))


# In[34]:


# Probabilidade do modelo

naive_bayes_predict = model_naive_bayes.predict_proba(freq).round(2)
naive_bayes_predict


# In[35]:


# Previsão do modelo

naive_bayes_predict = model_naive_bayes.predict(freq)
naive_bayes_predict


# In[36]:


# Accuracy do modelo
from sklearn.metrics import accuracy_score

accuracy_naive_bayes_multinomialNB = accuracy_score(teste, naive_bayes_predict)
print("Accuracy - Naive bayes multinomialNB: %.2f" % (accuracy_naive_bayes_multinomialNB * 100))


# In[37]:


# Confusion matrix
from sklearn.metrics import confusion_matrix

matrix_confusion_1 = confusion_matrix(teste, naive_bayes_predict)

plt.figure(figsize=(18.2, 10))
ax= plt.subplot()
sns.heatmap(matrix_confusion_1, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Naive bayes'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[38]:


# Classification report do modelo
from sklearn.metrics import classification_report

classification = classification_report(teste, naive_bayes_predict)
print("Modelo 01 - Naive bayes multinomialNB")
print()
print(classification)


# # Modelo 02 - Decision Tree

# In[39]:


get_ipython().run_cell_magic('time', '', '\n# Modelo machine learning - 2\n\n# Importação da biblioteca sklearn o modelo Decision Tree Classifier\nfrom sklearn.tree import DecisionTreeClassifier \n\n# Nome do algoritmo M.L\nmodel_dtc = DecisionTreeClassifier(max_depth = 5, random_state = 0)\n\n# Treinamento do modelo\nmodel_dtc_fit = model_dtc.fit(freq, teste)\n\n# Score do modelo\nmodel_dtc_score = model_dtc.score(freq, teste)\nprint("Modelo - Decision Tree Classifier: %.2f" % (model_dtc_score * 100))')


# In[40]:


# Previsão do modelo 
modelo_arvore_cla_1_predict = model_dtc.predict(freq)
modelo_arvore_cla_1_predict


# In[41]:


# Gráfico da ávore
from sklearn import tree

fig, ax = plt.subplots(figsize=(65.5, 45), facecolor = "w")

tree.plot_tree(model_dtc, 
               ax = ax, 
               fontsize = 25.8, 
               rounded = True, 
               filled = True, 
               class_names = ["Positivo", 
                              "Negativo", 
                              "Neutral", 
                              "Extremely Positive", 
                              "Extremely Negative"])
plt.show()


# In[42]:


# Accuracy score
acuracia_decision_tree = accuracy_score(teste, modelo_arvore_cla_1_predict)

print("Accuracy - Decision Tree: %.2f" % (acuracia_decision_tree * 100))


# In[43]:


# Confusion matrix
matrix_confusion_2 = confusion_matrix(teste, modelo_arvore_cla_1_predict)

plt.figure(figsize=(18.2, 10))
ax= plt.subplot()
sns.heatmap(matrix_confusion_2, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Decision tree'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[44]:


# Classification report
class_report = classification_report(teste, modelo_arvore_cla_1_predict)

print("Modelo - Decision Tree")
print("\n")
print(class_report)


# # Modelo 03 - Regressão logistica

# In[45]:


get_ipython().run_cell_magic('time', '', '\n# Importação da biblioteca sklearn o modelo Logistic Regression\nfrom sklearn.linear_model import LogisticRegression\n\n# Nome do algoritmo M.L\nmodel_regression_logistic = LogisticRegression()\n\n# Treinamento do modelo\nmodel_regression_logistic_fit = model_regression_logistic.fit(freq, teste)\n\n# Score do modelo\nmodel_regression_logistic_score = model_regression_logistic.score(freq, teste)\nprint("Modelo - Regressão logistica: %.2f" % (model_regression_logistic_score * 100))')


# In[46]:


# Previsão do modelo
model_regression_logistic_pred = model_regression_logistic.predict(freq)
model_regression_logistic_pred


# In[47]:


# Accuracy score
acuracia_Logistic_Regression = accuracy_score(teste, model_regression_logistic_pred)

print("Accuracy - Decision Tree: %.2f" % (acuracia_Logistic_Regression * 100))


# In[48]:


# Confusion matrix
matrix_confusion_3 = confusion_matrix(teste, model_regression_logistic_pred)

plt.figure(figsize=(18.2, 10))
ax= plt.subplot()
sns.heatmap(matrix_confusion_3, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Decision tree'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[49]:


# Classification report

classification = classification_report(teste, model_regression_logistic_pred)
print("Modelo 03 - Regressão logistica")
print()
print(classification)


# # Modelo 04 - K-NN

# In[50]:


get_ipython().run_cell_magic('time', '', '# Modelo machine learning - 4 KNN\n\n# Importação da biblioteca sklearn o K-NN\nfrom sklearn.neighbors import KNeighborsClassifier\n\n# Nome do algoritmo M.L\nmodel_knn = KNeighborsClassifier()\n\n# Treinamento do modelo\nmodel_knn_fit = model_knn.fit(freq, teste)\n\n# Score do modelo\nmodel_knn_score = model_knn.score(freq, teste)\nprint("Modelo - K-NN: %.2f" % (model_knn_score * 100))')


# In[51]:


# Previsão do modelo do k-nn

model_knn_pred = model_knn.predict(freq)
model_knn_pred


# In[52]:


# Accuracy modelo KNN
accuracy_knn = accuracy_score(teste, model_knn_pred)

print("Acurácia - K-NN: %.2f" % (accuracy_knn * 100))


# In[53]:


# Matrix confusion 
matrix_confusion_3 = confusion_matrix(teste, model_knn_pred)

plt.figure(figsize=(18.2, 10))
ax= plt.subplot()
sns.heatmap(matrix_confusion_3, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[54]:


# Classification report

classification = classification_report(teste, model_knn_pred)
print("Modelo 04 - K-NN")
print()
print(classification)


# # Modelo 05 - Random forest

# In[55]:


get_ipython().run_cell_magic('time', '', '# Modelo machine learning - 5\n\n# Importação da biblioteca sklearn o modelo random forest\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Importação da biblioteca sklearn o modelo Random Forest\nmodel_random_forest = RandomForestClassifier(max_depth = 2, random_state = 0) # max_depth - determinando total de árvore, random_state 0\n\n# Treinamento do modelo\nmodel_random_forest_fit = model_random_forest.fit(freq, teste) # Dados de treino, teste de x, y\n\n# Score do modelo\nmodel_random_forest_score = model_random_forest.score(freq, teste) # Valor da Accuracy do algoritmo \n\nprint("Score - Modelo random forest: %.2f" % (model_random_forest_score * 100))')


# In[56]:


# Previsão do modelo
model_random_forest_regressor_pred = model_random_forest.predict(freq)
model_random_forest_regressor_pred


# In[57]:


# Accuracy_score modelo

accuracy_random_forest = accuracy_score(teste, model_random_forest_regressor_pred)
print("Accuracy - Random forest: %.2f" % (accuracy_random_forest * 100))


# In[58]:


# Confusion matrix

matrix_confusion_4 = confusion_matrix(teste, model_random_forest_regressor_pred)
plt.figure(figsize=(18.2, 10))
ax = plt.subplot()
sns.heatmap(matrix_confusion_4, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma');

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Decision tree'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[59]:


# Classification report
from sklearn.metrics import classification_report

classification = classification_report(teste, model_random_forest_regressor_pred)
print("Modelo 05 - Random forest")
print()
print(classification)


# # Modelo 06 - XGBOOST

# In[60]:


# Modelo 06 - XGBosst 

# Importação da biblioteca sklearn o modelo XGBoost
from xgboost import XGBClassifier

# Nome do algoritmo M.L
xgb = XGBClassifier()

# Treinamento do modelo
xgb_fit = xgb.fit(freq, teste)

# Score do modelo
xgb_score = xgb.score(freq, teste)
print("Modelo - XGBoost: %.2f" % (xgb_score * 100))


# In[61]:


# Previsão do modelo

xgb_pred = xgb.predict(freq)
xgb_pred


# In[62]:


# Accuracy score do modelo

accuracy_XGBoost = accuracy_score(teste, xgb_pred)
print("Accuracy - XGBoost: %.2f" % (accuracy_XGBoost * 100))


# In[63]:


# Confusion matrix
matrix_confusion_4 = confusion_matrix(teste, xgb_pred)

plt.figure(figsize=(18.2, 10))
ax = plt.subplot()
sns.heatmap(matrix_confusion_4, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma');

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Decision tree'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutral", "Extremely Positive", "Extremely Negative"]);


# In[64]:


classification = classification_report(teste, xgb_pred)
print("Modelo 05 - Random forest")
print()
print(classification)


# In[65]:


# Resultados - Modelos machine learning

modelos = pd.DataFrame({
    
    "Models" :["Regressão logistica", 
                "K-NN", 
                "Random Forest", 
                "Decision Tree",
                "Naive Bayes",
                "XGBoost"],

    "Acurácia" :[accuracy_naive_bayes_multinomialNB, 
                      acuracia_decision_tree, 
                      acuracia_Logistic_Regression, 
                      accuracy_knn,
                      accuracy_random_forest,
                      accuracy_XGBoost]})

modelos.sort_values(by = "Acurácia", ascending = False)


# In[66]:


## Salvando modelo M.L PLN

import pickle
 
with open('naive_bayes_predict.pkl', 'wb') as file:
    pickle.dump(naive_bayes_predict, file)

with open('modelo_arvore_cla_1_predict.pkl', 'wb') as file:
    pickle.dump(modelo_arvore_cla_1_predict, file)

with open('model_regression_logistic_pred.pkl', 'wb') as file:
    pickle.dump(model_regression_logistic_pred, file)

with open('model_knn_pred.pkl', 'wb') as file:
    pickle.dump(model_knn_pred, file)

with open('model_random_forest_regressor_pred.pkl', 'wb') as file:
    pickle.dump(model_random_forest_regressor_pred, file)

with open('xgb_pred.pkl', 'wb') as file:
    pickle.dump(xgb_pred, file)


# # Análise sentimento nas frases com vader

# In[68]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

model_sid = SentimentIntensityAnalyzer()
model_sid.polarity_scores(df.iloc[0]['OriginalTweet'])


# In[69]:


df['scores'] = df['OriginalTweet'].apply(lambda review:model_sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d:d['compound'])
df['score'] = df['compound'].apply(lambda score: 'pos' if score >=0 else 'neg')

df.head()


# In[ ]:




