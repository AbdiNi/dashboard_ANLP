import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H2(' Please select wich page you want to visualize : '),
    
    dcc.Link(' visualisation des données ', href='/DataViz'),
    html.Br(),  #saut de ligne
    dcc.Link(' Evaluation des classifieurs ', href='/EVALUATION')
])


