import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from app import app
import Data as dt
import pandas as pd
from time import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords 

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")


pipe1 = Pipeline([('vect', CountVectorizer()), ('sgd', SGDClassifier()),])
pipe2 = Pipeline([('vect', CountVectorizer()),('svm', SVC()),])
pipe3 = Pipeline([('vect', CountVectorizer()), ('logreg', LogisticRegression()),])
pipe4 = Pipeline([('vect', CountVectorizer()),('KNN', KNeighborsClassifier(n_neighbors=4)),]) 
pipe5 = Pipeline([('vect', CountVectorizer()),('DTC', DecisionTreeClassifier(max_depth=4)),])

def run_pipes(pipes) :
    
    df = pd.DataFrame(columns=['model', 'fitting_time', 'accuracy',  'precision', 'recall'])
    
    for pipe in pipes : 
        start = time()
        pipe.fit(X_train, y_train)
        fit_time = time() - start
        y_pred =pipe.predict(X_test)
        
        acc = metrics.accuracy_score(y_test, y_pred) 
        pre = metrics.precision_score(y_test,y_pred,average ='macro', zero_division=0)
        rec = metrics.recall_score(y_test,y_pred, average ='macro', zero_division=0)
        
        df = df.append({'model':pipe.steps[1][0], 'fitting_time':round( fit_time, 2), 
                        'accuracy': round(acc,2), 'precision':round(pre,2),
                        'recall': round(rec,2)},ignore_index=True)
    return df


print("1")

X = dt.corpus1
y = dt.targets1
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

df_eval = run_pipes(pipes= [pipe1,  pipe3, pipe4,pipe5])

print("2")
#matrice de confusion
y_predc =pipe3.predict(X)
cm1 = confusion_matrix(y, y_predc)

print("3")

new_index = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
list_labels = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
df_conf= pd.DataFrame(columns=('anger', 'fear', 'happy', 'love', 'sadness', 'surprise', 'prediction souhaité'))
for i in range(cm1.shape[0]):
    list_pred=[cm1[i,0], cm1[i,1], cm1[i,2],cm1[i,3],cm1[i,4],cm1[i,5]]
    max_value = max(list_pred)
    m_index = list_pred.index(max_value)
    df_conf.loc[new_index[i]] = [cm1[i,0], cm1[i,1], cm1[i,2],cm1[i,3],cm1[i,4],cm1[i,5], list_labels[m_index]]


# Evaluation
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_eval.model, y=df_eval.accuracy, name="accuracy",  line_shape='linear'))
fig.add_trace(go.Scatter(x=df_eval.model, y=df_eval.precision, name="precision",  line_shape='linear'))
fig.add_trace(go.Scatter(x=df_eval.model, y=df_eval.recall, name="recall",  line_shape='linear'))

print("4")

layout = html.Div([
    dcc.Link('Visualiser les données ', href='/DataViz'),
    dbc.Alert("Evaluation des classifieurs", color="danger"),
  
    html.H2(' 1- Tableau d\'Evaluation entre 5 models sur le premier jeu de données'),
    html.Br(),
    dbc.Table.from_dataframe(df_eval, striped=True, bordered=True, hover=True),

    
    html.H2(' 2- Matrice de confusion (model = sgd) '),
    html.Br(),
    dbc.Table.from_dataframe(df_conf, striped=True, bordered=True, hover=True),
   

    html.H2(' 3- Comparatif des classifieurs du prememier jeu de données '),
    dcc.Graph(figure=fig),
   
    html.Div(id='app-2-display-value'),
    


    html.H6("Prédire un Text : "),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    html.Br(),
    html.Div(id='my-output'),
])


@app.callback(
    Output('app-2-display-value', 'children'),
    [Input('app-2-dropdown', 'value')])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)
def display_value(value):
    return 'You have selected "{}"'.format(value)
