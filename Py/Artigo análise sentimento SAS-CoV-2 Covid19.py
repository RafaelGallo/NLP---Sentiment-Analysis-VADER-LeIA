#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis VADER - Covid19 artigos

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


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


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


# Configuração para os gráficos largura e layout dos graficos

plt.style.use('fivethirtyeight')


# # Base dados

# In[6]:


df_1 = pd.read_csv("train.csv")
df_2 = pd.read_csv("test.csv")

df_1


# In[7]:


# Exibido 5 primeiros dados
df_1.head()


# In[8]:


# Exibido 5 últimos dados 
df_1.tail(5)


# In[9]:


# Número de linhas e colunas
df_1.shape


# In[10]:


# Verificando informações das variaveis
df_1.info()


# In[11]:


# Tipo dos dados
df_1.dtypes


# In[12]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(df_1.shape[0]))
print("Números de colunas: {}" .format(df_1.shape[1]))


# In[13]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", df_1.isnull().sum().values.sum())


# # Análise de dados

# In[17]:


# Removendo dados nulos

df_1.dropna(inplace=True)
df_2.dropna(inplace=True)


# In[14]:


# Verificando base
df_1.value_counts()


# In[16]:


# Verificando o total de sentimentos
df_1["ABSTRACT"].value_counts()


# In[21]:


# Nuvem de palavras
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

todos_palavras = ' '.join([texto for texto in df_1["ABSTRACT"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[23]:


todos_palavras = ' '.join([texto for texto in df_1["TITLE"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# # Análise sentimento nas frases

# In[25]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

model_sid = SentimentIntensityAnalyzer()
model_sid.polarity_scores(df_1.iloc[0]['ABSTRACT'])


# In[27]:


df_1['scores'] = df_1['ABSTRACT'].apply(lambda review:model_sid.polarity_scores(review))
df_1['compound'] = df_1['scores'].apply(lambda d:d['compound'])
df_1['score'] = df_1['compound'].apply(lambda score: 'pos' if score >=0 else 'neg')

df_1.head()


# In[33]:


plt.figure(figsize=(15.5, 10))

plt.title("Sentimentos nos textos")
sns.countplot(df_1["score"])
plt.xlabel("Score")
plt.ylabel("Total")


# In[39]:


plt.figure(figsize=(15.5, 10))

sns.histplot(df_1["compound"])
plt.title("Comportamento dos sentimentos")
plt.xlabel("Comportamento")
plt.ylabel("Total")


# In[42]:


# Nuvem de palavras
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

todos_palavras = ' '.join([texto for texto in df_1["TITLE"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[ ]:




