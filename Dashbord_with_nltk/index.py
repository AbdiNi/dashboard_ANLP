from app import server
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from Apps import app0, app1, app2


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    
    if pathname == '/DataViz':
        return app1.layout
    elif pathname == '/EVALUATION':
        return app2.layout
    else:
        return app0.layout


if __name__ == '__main__':
    app.run_server(debug=True)
