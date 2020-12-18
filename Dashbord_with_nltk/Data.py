import plotly.express as px
import plotly.graph_objs as go
import pandas as pd 
import numpy as np
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

#****************** Récupération des données CSV ************************#

datak = pd.read_csv('/home/nidhal/Projets_Simplon/Brief5_ANLP_Emotions/Emotion_final.csv')
dataw = pd.read_csv('/home/nidhal/Projets_Simplon/Brief5_ANLP_Emotions/text_emotion.csv')

df1 = datak.iloc[:5,:]  # 8lines contains "NaN"
df2 = dataw.iloc[:5,:]

#nombre d'observations
n = datak.shape[0]
#nombre de variables
p = datak.shape[1]

value_d = 0

targets1 = list(datak["Emotion"])
corpus1 = list(datak["Text"])

targets2 = list(dataw["sentiment"])
corpus2 = list(dataw["content"])


#**************** Histograms and Pies with plotly express ******************

# Histogramme 1
Hist1 = px.histogram(x=targets1, nbins=4, width=1000, height=500).update_xaxes(categoryorder = 'total descending')
Hist1.update_layout( title="Histogramme des sentiments ",
    xaxis_title="Sentiments",
    yaxis_title="N_Fois")
    

# Histogramme 2
Hist2 = px.histogram(x=targets2, nbins=13, width=1000, height=500).update_xaxes(categoryorder = 'total descending')
Hist2.update_layout( title="Histogramme des sentiments ",
    xaxis_title="Sentiments",
    yaxis_title="N_Fois")

# Pie 1
list_labels = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
list_freq = [0,0,0,0,0,0]
for i in range(datak.shape[0]):
    ind = list_labels.index(datak["Emotion"][i])
    list_freq[ind] += 1 
intermediate_dictionary = {'Labels':list_labels, 'Freq':list_freq}
Freq_df = pd.DataFrame(intermediate_dictionary)
Pie1 = px.pie(Freq_df, values='Freq', names='Labels')
Pie1.update_layout( title="Pie des sentiments ")

# Pie 2

list_labels2 = list(np.unique(dataw.sentiment))
list_freq2 = [0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(dataw.shape[0]):
    ind = list_labels2.index(dataw["sentiment"][i])
    list_freq2[ind] += 1 
intermediate_dictionary = {'Labels':list_labels2, 'Freq':list_freq2}
Freq_df2 = pd.DataFrame(intermediate_dictionary)
Pie2 = px.pie(Freq_df2, values='Freq', names='Labels')
Pie2.update_layout( title="Pie des sentiments ")





#**************** Get and plot top n_grams ******************

def get_top_ngram(corpus, dim , n=None):
    vec = CountVectorizer(ngram_range=(dim,dim )).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
 
if(value_d>0) : 
    print('ngrams')
    common_words = get_top_ngram(corpus1, value_d, 5)
    df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])


# #******************* Manually preprocessing ********************

def tokenize(corpus, n):
    #Tokenization
    words=[]
    for i in range(len(corpus)):
        # tokens = nltk.word_tokenize(corpus[i])
        tokens = corpus[i].split()
        words.append(tokens)
    flat_words = [item for sublist in words for item in sublist]
    
    stopW = stopwords.words('english')
    stopW.extend(set(string.punctuation))
    tokens_without_stopwords = [x for x in flat_words if x not in stopW]
    n_grams = ngrams(tokens_without_stopwords,n)
    
    return n_grams


def plot_fig(ngrams):
    #ploting with seaborn
    sns.set(rc={'figure.figsize' : (11,4)})
    sns.set_style('darkgrid')
    # nlp_words = nltk.FreqDist(ngrams)
    #nlp_words.plot(20)
    
def get_top_words_or_ngrams():
    tokens_ngrams = tokenize(corpus1, n)  
    
 

