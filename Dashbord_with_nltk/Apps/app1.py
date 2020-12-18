import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import Data as dt
import dash_table
import dash
from dash.dependencies import Input, Output
import pandas as pd

from app import app
import plotly.graph_objs as go


from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords





def generate_table(dataframe, max_rows=50):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# def get_top_ngram(corpus, dim , n=None):
    
#     #df = pd.DataFrame(columns=['Ngrams', 'dimension', 'Occurance'])

#     vec = CountVectorizer(ngram_range=(dim, dim), stop_words=stopwords.words("english")).fit(corpus)
#     bag_of_words = vec.transform(corpus)
#     sum_words = bag_of_words.sum(axis=0) 
#     words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#     words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
#     print(words_freq[:n])
#     dfw =  pd.DataFrame(words_freq, columns=["A", "B"])
    
#     return dfw 
#     #return words_freq[:n]

# dfw1 = get_top_ngram(dt.corpus1, 3 , 5 )
# print(dfw1)


layout = html.Div([
    dcc.Link('Evaluer les classifieurs ', href='/EVALUATION'),
    html.Div(id='app-1-display-value'),
    dbc.Alert(" Visualisation des données ", color="primary"),
    
    html.H2(' 1- Importation de la première dataset '),
    html.Br(),
    dbc.Table.from_dataframe(dt.df1, striped=True, bordered=True, hover=True),
    
    html.H2(' 2- Pie des Émotions (Premiere Dataset) '),
    
    dcc.Graph( figure=dt.Pie1 ),

    html.H2(' 3- Classement des Émotions (Premiere Dataset) '),
    dcc.Graph( figure=dt.Hist1 ),

    html.H2(' 4- Importation de deuxième dataset '),
    dbc.Table.from_dataframe(dt.df2, striped=True, bordered=True, hover=True),
    html.Br(),
    
    html.H2(' 5- Pie des Émotions (Deuxième Dataset) '),
    dcc.Graph(figure=dt.Pie2 ),

    html.H2(' 6- Histogramme des Émotions (Deuxième Dataset) '),
    dcc.Graph(  figure=dt.Hist2 ),

    html.H2(' 7- Ngrams dataframe '),
    #dbc.Table.from_dataframe(dt.df3, striped=True, bordered=True, hover=True),
    html.Div(["N_Grams dimension : ",
        html.Div(dcc.Input(id='input-on-submit', type='number')),
        html.Button('Submit', id='submit-val', n_clicks=0),
        html.Div(id='container-button-basic',children='Enter a value and press submit'),
    html.Br(),
    ]),

    
    
])

@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])


def update_output(n_clicks, value):
    dt.value_d = value
    print( 'The input value was "{}"'.format(value) )
    
    return


    
@app.callback(
    Output('app-1-display-value', 'children'),
    Input('app-1-dropdown', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)