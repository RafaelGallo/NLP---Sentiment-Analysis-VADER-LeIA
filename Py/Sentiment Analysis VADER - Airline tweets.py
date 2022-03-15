#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis VADER - Airline

# <br>
# 
# <img src = "https://img.freepik.com/vetores-gratis/ilustracao-do-conceito-de-analise-de-sentimento_114360-5182.jpg?t=st=1647317146~exp=1647317746~hmac=9b747d107021265b6937bf4bd8dbe3a0d2ea04fdbfe1afc6ce773add5d463537&w=740">
# 
# <br>
# 
# Um trabalho de análise de sentimentos sobre os problemas de cada grande companhia aérea dos EUA. Os dados do Twitter foram extraídos de fevereiro de 2015 e os colaboradores foram solicitados a classificar primeiro os tweets positivos, negativos e neutros, seguidos pela categorização de motivos negativos (como "vôo atrasado" ou "serviço rude").
# 
# Por exemplo, ele contém se o sentimento dos tweets neste conjunto foi positivo, neutro ou negativo para seis companhias aéreas dos EUA:
# 
# As informações dos principais atributos para este projeto são as seguintes;
# 
# * **`alirline_sentiment`** : classificação de sentimento. (positiva, neutra e negativa)
# * **`negativereason`** : Motivo selecionado para a opinião negativa
# * **`airline`** : Nome de 6 companhias aéreas dos EUA ('Delta', 'United', 'Southwest', 'US Airways', 'Virgin America', 'American')
# * **`texto`** : Opinião do cliente
# 
# <br>

# In[21]:


# Versão do python

from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[22]:


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


# In[23]:


# NLTK para NLP
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download("stopwords")
nltk.download('punkt')


# In[24]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# In[25]:


# Configuração para os gráficos largura e layout dos graficos

plt.style.use('fivethirtyeight')


# # Base dados

# In[26]:


df = pd.read_csv("Tweets.csv")
df


# In[27]:


# Exibido 5 primeiros dados
df.head()


# In[28]:


# Exibido 5 últimos dados 
df.tail(5)


# In[29]:


# Número de linhas e colunas
df.shape


# In[30]:


# Tipo dos dados
df.dtypes


# In[31]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(df.shape[0]))
print("Números de colunas: {}" .format(df.shape[1]))


# In[32]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", df.isnull().sum().values.sum())


# In[33]:


# Verificando o total de sentimentos
df["airline_sentiment"].value_counts()


# # Análise de dados

# In[34]:


# Verificando o total de sentimentos

df_x = df.airline_sentiment.value_counts().sort_values()
df_x


# In[35]:


# Ajutando os dados de Companhias áreas
da = df.groupby(['airline', 'airline_sentiment'])['airline_sentiment'].count().unstack()
da


# In[36]:


# Print dos dados negativos

print(df.negativereason.unique())
print(df[df.negativereason == '']['airline_sentiment'].unique())
print(df[df.negativereason != '']['airline_sentiment'].unique())


# In[37]:


# Fazendo um display nos dados

df_neg = df[df.airline_sentiment == 'negative']
display(df_neg.shape, df_neg.head())


# In[38]:


plt.figure(figsize=(20.5, 15))

plt.title("Análise sentimento das frases")
ax = sns.countplot(df["airline_sentiment"])
plt.xlabel("Sentimento")
plt.ylabel("Total")


# In[39]:


plt.figure(figsize=(20.5, 15))

ax = sns.countplot(data = df, y = 'airline',
                   order = df.airline.value_counts().index)
plt.title('Companhias aéreas')
plt.xlabel("Airlines")
plt.ylabel("Total")
plt.show()


# In[42]:


plt.figure(figsize=(20.5, 15))

ax = plt.pie(x = df_x, labels=df_x.index, autopct = '%1.1f%%', explode = [0.03, 0.03, 0.08])
plt.title('Gráfico de pizza - Sentimentos')

plt.show()


# In[43]:


fig, axes = plt.subplots(2, 3, figsize = (15, 8))
axes = axes.flatten()

for i, ax in zip(range(0, 6), axes):
    temp = da.iloc[i]    
    ax.pie(x = temp, labels = temp.index, autopct = '%1.1f%%', explode = [0.08, 0.03, 0.03])
    ax.set_title(f"{da.index[i]}:{format(da.values[i].sum(),',')}")

plt.suptitle("Sentimentos", fontsize = 25)    
plt.show()


# In[44]:


plt.figure(figsize=(20.5, 15))

plt.title("Airline")
ax = sns.countplot(df["airline"])
plt.xlabel("airline x")
plt.ylabel("Total")


# In[45]:


df.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar',figsize=(20.5, 15))


# In[46]:


plt.figure(figsize=(20.5, 15))

sns.histplot(df["negativereason_confidence"])
plt.title("Confiança de negatividade")
plt.xlabel("Negativereason confidence")
plt.ylabel("Total")


# In[47]:


plt.figure(figsize=(8,5))
ax = sns.countplot(data = df_neg, y = 'negativereason',
                   palette='Set2',
                   order = df_neg.negativereason.value_counts().index)
ax.set_title('Contagem por Razão Negativa')

plt.show()


# In[48]:


fig, axes = plt.subplots(6, 1, figsize=(8, 25), sharex=True)
axes = axes.flatten()
names = df_neg['airline'].unique()

for name, n in zip(names, axes):
    ax = sns.countplot(data = df_neg[df_neg.airline==name], y = 'negativereason',
                       palette='Set2',
                       order = df_neg[df_neg.airline==name].negativereason.value_counts().index, ax = n)
    
    ax.set_title(f"{name}: {format(len(df_neg[df_neg.airline==name]),',')}")
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.suptitle("Contagem por motivo negativo", fontsize = 25)
plt.show()


# In[49]:


plt.figure(figsize=(20,8))
ax = sns.countplot(data = df_neg, x = 'negativereason',
                   palette='Set2',
                   order = df_neg.negativereason.value_counts().index, hue = 'airline')

#ax.bar_label(ax.containers[0])
ax.set_title('Count per NegativeReason')
plt.xticks(rotation=45)
plt.show()


# In[50]:


# Nuvem de palavras
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

todos_palavras = ' '.join([texto for texto in df["text"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[51]:


todos_palavras = ' '.join([texto for texto in df["airline"]])
nuvem_palavras = WordCloud(width = 800, height = 500, max_font_size = 110,
                          collocations = False).generate(todos_palavras)

plt.figure(figsize= (10, 7))
plt.imshow(nuvem_palavras, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# # Vader Sentiment

# In[52]:


# Verificando as colunas do dataset

df.columns


# In[53]:


# Definindo os dados para análise
df = df[['airline_sentiment', 'airline','text' ]]
df.head()


# In[54]:


# Definindo os dados para análise de dados da Companhia - United

df = df[df["airline"]=="United"]
df = df.reset_index(drop = True)

print(len(df))
df.head()


# In[55]:


# Análise de sentimento nas frases

# Importando a biblioteca vader sentiment, sentiment intensity analyzer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Modelo da análise de sentimento
model = SentimentIntensityAnalyzer()

# Função das frases
def data_model(df):
    
    # Modelo polaridade e score das frases
    model_set = model.polarity_scores(df) 
    
    # Print análise de sentimento
    print("{:-<40} {}".format(df, str(model_set)))

# Verificando sentimento
data_model("O voo da United foi uma experiência ruim") 


# ### Calculando a pontuação para cada tweet no dataframe/conjunto de dados

# In[56]:


get_ipython().run_line_magic('time', '')

x = 0

# Modelo
model_1 = []

# Lendo os dados do score
while(x<len(df)):
    
    # Modelo
    y = model.polarity_scores(df.iloc[x]['text'])
    
    # Armazenando no modelo
    model_1.append(y["compound"])
    x = x + 1
    
# Adicionando no array 
model_1 = np.array(model_1)

# Lendo o total de frases do modelo
len(model_1)


# In[57]:


# Score do modelo

df["Score_vader"] = model_1
df.head(20)


# In[60]:


get_ipython().run_line_magic('time', '')

x = 0

# Adicionando uma list
pred = []

# Definindo sentimento positivo
while(x<len(df)):
    if ((df.iloc[x]["Score_vader"] >= 0.7)):
        pred.append("Positivo")
        x = x + 1
    
    # Definindo sentimento neutro
    elif((df.iloc[x]["Score_vader"] > 0) & (df.iloc[x]["Score_vader"] < 0.7)):
        pred.append("Neutro")
        x = x + 1
    
    # Definindo sentimento negativo
    elif ((df.iloc[x]["Score_vader"] <= 0)):
        pred.append("Negativo")
        x = x + 1


# In[63]:


# Visualizando a análise de sentimento

df["Previsão"] = pred
len(df['Previsão'])


# In[62]:


# Visualizando total de sentimentos
df.head(50)


# In[ ]:





# In[ ]:





# In[ ]:




