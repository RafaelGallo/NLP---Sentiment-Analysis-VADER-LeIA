# NLP---Sentiment-Analysis-VADER

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/ilustracao-do-conceito-de-extracao-de-dados_114360-4906.jpg?w=740)

# Descrição do projeto
Projeto de machine learning NLP - Análise de sentimento utlizando a biblioteca NLTK VADER para análise de sentimentos nas frases. E possível indentificar pessoas sentimentos negativos, positivos, neutra num determinando caso.
Em NLP muito usado em ecommerce, rede social em RS e possível localizar pessoas com depressão ou tipo.
Nesse projeto fiz dois modelos um covid19, Companhias aéreas américanas. 
Logo logo tem mais projetos em NLP.

## Stack utilizada

**Programação** Python

**Machine learning**: Scikit-learn

**Leitura CSV**: Pandas

**Análise de dados**: Seaborn, Matplotlib

**Modelo machine learning - Processo de linguagem natural**: NLTK, TextBlob

## Projetos - Sentiment Analysis VADER

| Projeto               | Link                                                |
| ----------------- | ---------------------------------------------------------------- |
| Sentiment Analysis - ARTIGO SAS-CoV2-Covid19|[Link](https://github.com/RafaelGallo/NLP---Sentiment-Analysis-VADER/blob/main/notebooks/Artigo%20an%C3%A1lise%20sentimento%20SAS-CoV-2%20Covid19.ipynb)|
|Sentiment Analysis - SAS-CoV2-Covid19|[Link](https://github.com/RafaelGallo/NLP---Sentiment-Analysis-VADER/blob/main/notebooks/Artigo%20an%C3%A1lise%20sentimento%20SAS-CoV-2%20Covid19.ipynb)|
| Sentiment Analysis - Airline |[Link](https://github.com/RafaelGallo/NLP---Sentiment-Analysis-VADER/blob/main/notebooks/Sentiment%20Analysis%20VADER%20-%20Airline%20tweets.ipynb)|

## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo Modelo PLN  

```
# Importação das bibliotecas de nlp

import nltk
import string
import re
import warnings

nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words("english") + list(string.punctuation)
stopwords

# Pre-processamento
numerical = []

for i in df.columns:
    if df[i].dtype != 'object':
        numerical.append(i)
        
def data_pre(data):
    df['Numero_words'] = df['Text'].apply(lambda x : len([x for x in x.split()]))
    df['Numero_stopwords'] = df['Text'].apply(lambda x : len([x for x in x.lower().split() if x in stopwords]))
    df['Numero_special_char'] = df['Text'].apply(lambda x : len([x for x in x.split() if x in '[\w\s]']))
    df['Numero_chars'] = df['Text'].apply(lambda x : len(''.join([x for x in x.split()])))
    df['Text'] = df['Text'].apply(lambda x : x.lower())
    df['Text'] = df['Text'].str.replace('[^\w\s]','')
    df['Text'] = df['Text'].apply(lambda x : ' '.join(x for x in x.split() if x not in stopwords))
    df['Text'] = df['Text'].apply(lambda x : ' '.join(x for x in x.split() if x.isdigit()==False))

    return data
df.head()

# Stemmer
from nltk.stem import PorterStemmer

PORTER_STEMMER = PorterStemmer()
df['Text'] = df['Text'].apply(lambda x : ' '.join(PORTER_STEMMER.stem(x) for x in x.split()))
most_common = nltk.FreqDist(' '.join(df['Text']).split()).most_common(2000)
print(most_common)

# Treino e teste
processed_features = df['Text']
labels = df['Sentimento']


# Dados de limpeza para modelo PLN
# Remove stop words: Removendo as stop words na base de dados

# Text stemming: Palavras derivacionalmente relacionadas com significados semelhantes, palavras para retornar documentos que contenham outra palavra no conjunto.
# Dados limpos: Limpeza na base de dados limpando dados de web com http e outros.
# Lemmatization: Em linguística é o processo de agrupar as formas flexionadas de uma palavra para que possam ser analisadas como um único item, identificado pelo lema da palavra , ou forma de dicionário.
# Preprocessing: Pré - processamento da base de dados que serão ser para análise de dados.

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def remove_stop_words(instancia):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

def dados_limpos(instancia):
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instancia)


# Vectorizer NLP
# fidfVectorizer: Converta uma coleção de documentos brutos em uma matriz de recursos do TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_TFIDF = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
vectorizer_TFIDF_features = vectorizer_TFIDF.fit_transform(processed_features).toarray()
vectorizer_TFIDF_features

# Tokenização as palavras precisam ser codificadas como inteiros, 
# ou valores de ponto flutuante, para serem usadas como entradas para modelos machine learning.

from sklearn.feature_extraction.text import CountVectorizer

vetor = CountVectorizer(analyzer = "word", tokenizer = tweet_tokenizer.tokenize)
vetor_train = vetor.fit_transform(processed_features)
vetor_train.shape

# Modelo machine learning
# Modelo de regressão logistica 
from sklearn.linear_model import LogisticRegression

modelo_regression_logistic = LogisticRegression()
modelo_regression_logistic_fit = modelo_regression_logistic.fit(vetor_train, labels)
modelo_regression_logistic_score = modelo_regression_logistic.score(vetor_train, labels)

print("Model - Logistic Regression: %.2f" % (modelo_regression_logistic_score * 100))

# Previsão modelo com função predict de previsã das frases
modelo_regression_logistic_pred = modelo_regression_logistic.predict(vetor_train)
modelo_regression_logistic_pred

# Classification report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(labels, modelo_regression_logistic_pred))

# Accuracy do modelo
from sklearn import metrics
accuracy_regressao_logistica = metrics.accuracy_score(labels, modelo_regression_logistic_pred)
print("Accuracy model Logistic Regression: %.2f" % (accuracy_regressao_logistica * 100))

# Confusion matrix
matrix_1 = confusion_matrix(labels, modelo_regression_logistic_pred)

x = ["Negativo", "Neutro", "Positivo"]
y = ['Negativo', "Neutro", "Positivo"]

matrix = pd.DataFrame(matrix_1, columns=np.unique(y), index = np.unique(x))
matrix.index.name = 'Actual'
matrix.columns.name = 'Predicted'

plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
plt.title("Matrix confusion - Logistic Regression")
matrix = sns.heatmap(matrix, cmap = 'Paired', annot=True, annot_kws = {"size": 20}, fmt = "")
}
```

## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com
