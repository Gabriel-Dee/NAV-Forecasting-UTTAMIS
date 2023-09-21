from dash import Dash, html, dcc, Output, Input, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

data = pd.read_csv('../Data/NAV.csv')

app = Dash(__name__)
server = app.server

table_header_style = {
    'fontWeight': 'bold',
    'textShadow': '2px 2px 4px rgba(0,0,0,0.2)'
}

app.title = 'Net Asset Value Forecasting Dashboard'
app.layout = html.Div(style={'padding': '70px', 'paddingTop': '0px'}, children=[

    # Data Table
    html.H3(children='Data Table', style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='column',
        options=[{'label': column, 'value': column} for column in data.columns],
        value=data.columns,
        multi=True,
    ),
    dcc.Markdown('##'),
    dash_table.DataTable(
        id='data-table',
        columns=[
            {'name': column, 'id': column} for column in data.columns
        ],
        data=data.to_dict('records'),
        page_size=10,
        style_table={
            'textShadow': '2px 2px 4px rgba(0,0,0,0.4)',
        },
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center',
            'textTransform': 'capitalize',
            'backgroundColor': 'rgba(211, 211, 211, 0.7)'
        },
        style_data={
            'textAlign': 'center',
        },
    ),

    # Plot 1
    html.Hr(),
    html.H3(children='Average Net Asset Values Over Time', style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='schemes',
        options=[{'label': scheme, 'value': scheme} for scheme in data['Scheme Name'].unique()],
        value=[scheme for scheme in data['Scheme Name'].unique()],
        multi=True
    ),
    dcc.Graph(id='schemes-data', style={'height': '600px'}),

    # Plot 2
    html.Hr(),
    html.H3(children='Scheme Distribution Pie Chart', style={'textAlign': 'center', 'paddingTop': '10px',
                                                            'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='schemes-pie-dropdown',
        options=[{'label': scheme, 'value': scheme} for scheme in data['Scheme Name'].unique()],
        value=[scheme for scheme in data['Scheme Name'].unique()],
        multi=True
    ),
    dcc.Graph(id='schemes-pie', style={'height': '600px'}),

    # Plot 3
    html.Hr(),
    html.H3(children='Average NAV Trend Over Years Bar Graph', style={'textAlign': 'center', 'paddingTop': '10px',
                                                                      'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='year',
        options=[{'label': year, 'value': year} for year in data['Year'].unique()],
        value=[year for year in data['Year'].unique()],
        multi=True
    ),
    dcc.Graph(id='nav-trend', style={'height': '600px'}),

    # Plot 4
    html.Hr(),
    html.H3(children='Average NAV Trend Over Months Bar Graph', style={'textAlign': 'center', 'paddingTop': '10px',
                                                                        'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='month',
        options=[{'label': month, 'value': month} for month in data['Month'].unique()],
        value=[month for month in data['Month'].unique()],
        multi=True
    ),
    dcc.Graph(id='month-nav-trend', style={'height': '600px'}),

    # Plot 5
    html.Hr(),
    html.H3(children='Average Scheme Prices Over Month', style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='schemes-month',
        options=[{'label': scheme, 'value': scheme} for scheme in data['Scheme Name'].unique()],
        value=[scheme for scheme in data['Scheme Name'].unique()],
        multi=True
    ),
    dcc.Graph(id='schemes-month-data', style={'height': '600px'}),

    # Plot 6
    html.Hr(),
    html.H3(children='Net Asset Value Distribution vs Scheme Box-plot',
            style={'textAlign': 'center', 'paddingTop': '10px', 'fontFamily': 'sans-serif'}),
    dcc.Dropdown(
        id='scheme-box',
        options=[{'label': scheme, 'value': scheme} for scheme in data['Scheme Name'].unique()],
        value=[scheme for scheme in data['Scheme Name'].unique()],
        multi=True
    ),
    dcc.Graph(id='box-scheme', style={'height': '600px'}),
    html.Hr(),

])

#################################################################################################################
# Table callback
@app.callback(
    [Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('column', 'value')]
)
def update_table(selected_columns):
    filtered_data = data[selected_columns]
    columns = [{'name': column, 'id': column} for column in selected_columns]
    return filtered_data.to_dict('records'), columns

# Plot 1 callback
@app.callback(
    Output('schemes-data', 'figure'),
    [Input('schemes', 'value')]
)
def update_graph(selected_schemes):
    filtered_data = data[data['Scheme Name'].isin(selected_schemes)]
    fig = px.line(
        filtered_data, x='Year', y='Net Asset Value', color='Scheme Name', markers=True, line_shape='linear'
    )
    fig.update_layout(
        xaxis=dict(title='Year'),
        yaxis=dict(title='Average NAV'),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=50),
        title=f'Average NAV Trends for {", ".join(selected_schemes)}',
        title_x=0.5,
        title_y=0.96
    )
    return fig

# Plot 2 callback
@app.callback(
    Output('schemes-pie', 'figure'),
    [Input('schemes-pie-dropdown', 'value')]
)
def commodity_pie(selected_schemes):
    filtered_data = data[data['Scheme Name'].isin(selected_schemes)]
    fig = px.pie(
        filtered_data, names='Scheme Name', title='Schemes Distribution',
        labels={'Scheme Name': 'Scheme Name'}
    )
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05, 0.05, 0.05],
        marker=dict(line=dict(color='black', width=2))
    )
    fig.update_layout(
        margin=dict(t=40, b=30),
        title_x=0.5
    )
    return fig

# Plot 3 callback
@app.callback(
    Output('nav-trend', 'figure'),
    [Input('year', 'value')]
)
def price_trend(selected_years):
    filtered_data = data[data['Year'].isin(selected_years)]
    fig = px.bar(
        filtered_data, x='Year', y='Net Asset Value', color='Scheme Name',
        title='Average NAV of Schemes Over the Years',
        labels={'Year': 'Year', 'Net Asset Value': 'Average Net Asset Value'},
        barmode='group'
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=30),
        legend_title='Scheme',
        title_x=0.5,
        title_y=0.96
    )
    return fig

# Plot 4 callback
@app.callback(
    Output('month-nav-trend', 'figure'),
    [Input('month', 'value')]
)
def price_trend(selected_months):
    filtered_data = data[data['Month'].isin(selected_months)]
    fig = px.bar(
        filtered_data, x='Month', y='Net Asset Value', color='Scheme Name',
        title='Average Net Asset Values of Schemes Over the Months',
        labels={'Month': 'Month', 'Net Asset Value': 'Average Net Asset Value'},
        barmode='group'
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=30),
        legend_title='Scheme',
        title_x=0.5,
        title_y=0.96
    )
    return fig

# Plot 5 callback
@app.callback(
    Output('schemes-month-data', 'figure'),
    [Input('schemes-month', 'value')]
)
def update_graph(selected_schemes):
    filtered_data = data[data['Scheme Name'].isin(selected_schemes)]
    fig = px.line(
        filtered_data, x='Month', y='Net Asset Value', color='Scheme Name', markers=True, line_shape='linear'
    )
    fig.update_layout(
        xaxis=dict(title='Month'),
        yaxis=dict(title='Average NAV'),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=30),
        title=f'Average Net Asset Value Trends for {", ".join(selected_schemes)}',
        title_x=0.5,
        title_y=0.96
    )
    return fig

# Plot 6 callback
@app.callback(
    Output('box-scheme', 'figure'),
    [Input('scheme-box', 'value')]
)
def box_dist(selected_schemes):
    filtered_data = data[data['Scheme Name'].isin(selected_schemes)]
    fig = px.box(
        filtered_data, x='Scheme Name', y='Net Asset Value',
        title='Net Asset Value Distribution by Scheme Name',
        labels={'Scheme Name': 'Scheme Name', 'Net Asset Value': 'Net Asset Value'}
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=30),
        title_x=0.5,
        title_y=0.96
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, jupyter_mode="external")
