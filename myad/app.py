import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import DashProxy, ServersideOutputTransform
import numpy as np
import dash_bootstrap_components as dbc

# Load and clean data
df = pd.read_csv("Amazon Sales data.csv", encoding='latin-1')
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

# Convert date columns
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
df = df.dropna(subset=['Order Date'])

# Add derived columns
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month_name()
df['Processing Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce').fillna(0)
df['Total Cost'] = pd.to_numeric(df['Total Cost'], errors='coerce').fillna(0)
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce').fillna(0)
df['Unit Price'] = pd.to_numeric(df['Unit Price'], errors='coerce').fillna(0)

# Initialize Dash app with Bootstrap and suppress callback exceptions
app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], transforms=[ServersideOutputTransform()], suppress_callback_exceptions=True)

# Define page layouts
def home_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("📊 Amazon Sales Forecasting & Analytics: Trends, Insights, and Performance", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Predictions")),
                    dbc.CardBody([
                        html.P("Go to the Predictions page to see revenue predictions."),
                        dcc.Link('Go to Predictions', href='/predictions', className="btn btn-primary")
                    ])
                ], className="mb-3")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Insights")),
                    dbc.CardBody([
                        html.P("Visit the Insights page to uncover the reasons behind the trends and make informed decisions."),
                        dcc.Link('Go to Insights', href='/insights', className="btn btn-primary")
                    ])
                ], className="mb-3")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Trends")),
                    dbc.CardBody([
                        html.P("Go to the Trends page to explore patterns and changes in your data over time."),
                        dcc.Link('Go to Trends Page', href='/advanced-insights', className="btn btn-primary")
                    ])
                ], className="mb-3")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("ForeCasting")),
                    dbc.CardBody([
                        html.P("Go to the Forecasting page to explore in-depth data analysis and future projections."),
                        dcc.Link('Go to ForeCasting', href='/data-analytics', className="btn btn-primary")
                    ])
                ], className="mb-3")
            ], width=3)
        ])
    ], fluid=True)

def predictions_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("📈 Predicted Revenue", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Input(id='units-sold', type='number', placeholder='Units Sold', className="form-control"),
                dcc.Input(id='unit-price', type='number', placeholder='Unit Price', className="form-control mt-2"),
                dcc.Input(id='processing-time', type='number', placeholder='Processing Time', className="form-control mt-2"),
                html.Button('Predict Revenue', id='predict-button', n_clicks=0, className="btn btn-primary mt-2"),
                html.Div(id='prediction-output', className="mt-3")
            ], width=6)
        ], justify="center"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='predicted-revenue-graph'), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='revenue-cost-comparison'), width=6),
            dbc.Col(dcc.Graph(id='profit-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='units-sold-processing-time'), width=6),
            dbc.Col(dcc.Graph(id='unit-price-processing-time'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Link('Go back to Home', href='/', className="btn btn-secondary mt-3"), width=12)
        ])
    ], fluid=True)

def insights_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("🔍 Insights", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='dropdown1',
                options=[{'label': i, 'value': i} for i in df['Region'].dropna().unique()],
                multi=True,
                placeholder="Select Regions",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown2',
                options=[{'label': i, 'value': i} for i in df['Year'].unique()],
                multi=True,
                placeholder="Select Year",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown3',
                options=[{'label': i, 'value': i} for i in df['Country'].dropna().unique()],
                multi=True,
                placeholder="Select Country",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown4',
                options=[{'label': i, 'value': i} for i in df['Product'].dropna().unique()],
                multi=True,
                placeholder="Select Product",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown5',
                options=[{'label': i, 'value': i} for i in df['Order Priority'].dropna().unique()],
                multi=True,
                placeholder="Select Order Priority",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown6',
                options=[{'label': i, 'value': i} for i in df['Sales Channel'].dropna().unique()],
                multi=True,
                placeholder="Select Sales Channel",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown7',
                options=[{'label': i, 'value': i} for i in df['Month'].dropna().unique()],
                multi=True,
                placeholder="Select Month",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown8',
                options=[{'label': i, 'value': i} for i in df['Processing Time'].dropna().unique()],
                multi=True,
                placeholder="Select Processing Time",
                className="mb-3"
            ), width=3)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='total-sales-by-region'), width=6),
            dbc.Col(dcc.Graph(id='average-unit-price-cost'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='total-profit-by-country'), width=6),
            dbc.Col(dcc.Graph(id='sales-channel-order-priority'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='average-processing-time'), width=6),
            dbc.Col(dcc.Graph(id='total-sales-by-item-type'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='order-priority-by-region'), width=6),
            dbc.Col(dcc.Graph(id='unit-price-profit-correlation'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='seasonal-trends'), width=6),
            dbc.Col(dcc.Graph(id='units-sold-by-country'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Link('Go back to Home', href='/', className="btn btn-secondary mt-3"), width=12)
        ])
    ], fluid=True)

def advanced_insights_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("🔍 Trends", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='dropdown1-advanced',
                options=[{'label': i, 'value': i} for i in df['Region'].dropna().unique()],
                multi=True,
                placeholder="Select Regions",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown2-advanced',
                options=[{'label': i, 'value': i} for i in df['Year'].unique()],
                multi=True,
                placeholder="Select Year",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown3-advanced',
                options=[{'label': i, 'value': i} for i in df['Country'].dropna().unique()],
                multi=True,
                placeholder="Select Country",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown4-advanced',
                options=[{'label': i, 'value': i} for i in df['Product'].dropna().unique()],
                multi=True,
                placeholder="Select Product",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown5-advanced',
                options=[{'label': i, 'value': i} for i in df['Order Priority'].dropna().unique()],
                multi=True,
                placeholder="Select Order Priority",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown6-advanced',
                options=[{'label': i, 'value': i} for i in df['Sales Channel'].dropna().unique()],
                multi=True,
                placeholder="Select Sales Channel",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown7-advanced',
                options=[{'label': i, 'value': i} for i in df['Month'].dropna().unique()],
                multi=True,
                placeholder="Select Month",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown8-advanced',
                options=[{'label': i, 'value': i} for i in df['Processing Time'].dropna().unique()],
                multi=True,
                placeholder="Select Processing Time",
                className="mb-3"
            ), width=3)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='total-sales-revenue-by-country'), width=6),
            dbc.Col(dcc.Graph(id='unit-price-distribution'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='highest-average-unit-price'), width=6),
            dbc.Col(dcc.Graph(id='total-cost-outliers'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='total-profit-by-item-type'), width=6),
            dbc.Col(dcc.Graph(id='average-processing-time-by-country'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='highest-average-revenue-per-order'), width=6),
            dbc.Col(dcc.Graph(id='units-sold-profit-correlation'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='order-priority-by-item-type'), width=6),
            dbc.Col(dcc.Graph(id='order-date-trends'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Link('Go back to Home', href='/', className="btn btn-secondary mt-3"), width=12)
        ])
    ], fluid=True)

def data_analytics_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("📈 ForeCasting", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='dropdown1-analytics',
                options=[{'label': i, 'value': i} for i in df['Region'].dropna().unique()],
                multi=True,
                placeholder="Select Regions",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown2-analytics',
                options=[{'label': i, 'value': i} for i in df['Year'].unique()],
                multi=True,
                placeholder="Select Year",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown3-analytics',
                options=[{'label': i, 'value': i} for i in df['Country'].dropna().unique()],
                multi=True,
                placeholder="Select Country",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown4-analytics',
                options=[{'label': i, 'value': i} for i in df['Product'].dropna().unique()],
                multi=True,
                placeholder="Select Product",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown5-analytics',
                options=[{'label': i, 'value': i} for i in df['Order Priority'].dropna().unique()],
                multi=True,
                placeholder="Select Order Priority",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown6-analytics',
                options=[{'label': i, 'value': i} for i in df['Sales Channel'].dropna().unique()],
                multi=True,
                placeholder="Select Sales Channel",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown7-analytics',
                options=[{'label': i, 'value': i} for i in df['Month'].dropna().unique()],
                multi=True,
                placeholder="Select Month",
                className="mb-3"
            ), width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown8-analytics',
                options=[{'label': i, 'value': i} for i in df['Processing Time'].dropna().unique()],
                multi=True,
                placeholder="Select Processing Time",
                className="mb-3"
            ), width=3)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='analytics-graph1'), width=6),
            dbc.Col(dcc.Graph(id='analytics-graph2'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='analytics-graph3'), width=6),
            dbc.Col(dcc.Graph(id='analytics-graph4'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='analytics-graph5'), width=6),
            dbc.Col(dcc.Graph(id='analytics-graph6'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='analytics-graph7'), width=6),
            dbc.Col(dcc.Graph(id='analytics-graph8'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='analytics-graph9'), width=6),
            dbc.Col(dcc.Graph(id='analytics-graph10'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Link('Go back to Home', href='/', className="btn btn-secondary mt-3"), width=12)
        ])
    ], fluid=True)

# Main layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Callback to update page content
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/predictions':
        return predictions_layout()
    elif pathname == '/insights':
        return insights_layout()
    elif pathname == '/advanced-insights':
        return advanced_insights_layout()
    elif pathname == '/data-analytics':
        return data_analytics_layout()
    else:
        return home_layout()

# Callbacks for Predictions Graphs

# Callbacks for Predictions Graphs
@app.callback(
    [Output('predicted-revenue-graph', 'figure'),
     Output('revenue-cost-comparison', 'figure'),
     Output('profit-graph', 'figure'),
     Output('units-sold-processing-time', 'figure'),
     Output('unit-price-processing-time', 'figure'),
     Output('prediction-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('units-sold', 'value'),
     State('unit-price', 'value'),
     State('processing-time', 'value')]
)
def update_predictions_graphs(n_clicks, units_sold, unit_price, processing_time):
    if n_clicks > 0 and units_sold is not None and unit_price is not None:
        predicted_revenue = units_sold * unit_price
        total_cost = units_sold * unit_price * 0.5  # Example cost calculation
        profit = predicted_revenue - total_cost

        # Figures
        fig_predicted_revenue = px.bar(x=['Predicted Revenue'], y=[predicted_revenue], title="Predicted Revenue")
        fig_revenue_cost_comparison = px.bar(x=['Revenue', 'Cost'], y=[predicted_revenue, total_cost], title="Revenue vs Cost")
        fig_profit = px.bar(x=['Profit'], y=[profit], title="Predicted Profit")
        fig_units_sold_processing_time = px.scatter(x=[units_sold], y=[processing_time], labels={'x': 'Units Sold', 'y': 'Processing Time'}, title="Units Sold vs Processing Time")
        fig_unit_price_processing_time = px.scatter(x=[unit_price], y=[processing_time], labels={'x': 'Unit Price', 'y': 'Processing Time'}, title="Unit Price vs Processing Time")

        # Display the predicted revenue amount
        prediction_result = f"Predicted Revenue: ${predicted_revenue:.2f}"

        return (
            fig_predicted_revenue,
            fig_revenue_cost_comparison,
            fig_profit,
            fig_units_sold_processing_time,
            fig_unit_price_processing_time,
            prediction_result
        )
    return [px.scatter() for _ in range(5)] + [""]

# Prediction Page Layout
def predictions_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("📈 Predicted Revenue", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Input(id='units-sold', type='number', placeholder='Units Sold', className="form-control"),
                dcc.Input(id='unit-price', type='number', placeholder='Unit Price', className="form-control mt-2"),
                dcc.Input(id='processing-time', type='number', placeholder='Processing Time', className="form-control mt-2"),
                html.Button('Predict Revenue', id='predict-button', n_clicks=0, className="btn btn-primary mt-2"),
                html.Div(id='prediction-output', className="mt-3")
            ], width=6)
        ], justify="center"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='predicted-revenue-graph'), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='revenue-cost-comparison'), width=6),
            dbc.Col(dcc.Graph(id='profit-graph'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='units-sold-processing-time'), width=6),
            dbc.Col(dcc.Graph(id='unit-price-processing-time'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Link('Go back to Home', href='/', className="btn btn-secondary mt-3"), width=12)
        ])
    ], fluid=True)


# Callbacks for Insights Graphs
@app.callback(
    [Output('total-sales-by-region', 'figure'),
     Output('average-unit-price-cost', 'figure'),
     Output('total-profit-by-country', 'figure'),
     Output('sales-channel-order-priority', 'figure'),
     Output('average-processing-time', 'figure'),
     Output('total-sales-by-item-type', 'figure'),
     Output('order-priority-by-region', 'figure'),
     Output('unit-price-profit-correlation', 'figure'),
     Output('seasonal-trends', 'figure'),
     Output('units-sold-by-country', 'figure')],
    [Input('dropdown1', 'value'),
     Input('dropdown2', 'value'),
     Input('dropdown3', 'value'),
     Input('dropdown4', 'value'),
     Input('dropdown5', 'value'),
     Input('dropdown6', 'value'),
     Input('dropdown7', 'value'),
     Input('dropdown8', 'value')]
)
def update_insights_graphs(region, year, country, product, order_priority, sales_channel, month, processing_time):
    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df['Region'].isin(region)]
    if year:
        filtered_df = filtered_df[filtered_df['Year'].isin(year)]
    if country:
        filtered_df = filtered_df[filtered_df['Country'].isin(country)]
    if product:
        filtered_df = filtered_df[filtered_df['Product'].isin(product)]
    if order_priority:
        filtered_df = filtered_df[filtered_df['Order Priority'].isin(order_priority)]
    if sales_channel:
        filtered_df = filtered_df[filtered_df['Sales Channel'].isin(sales_channel)]
    if month:
        filtered_df = filtered_df[filtered_df['Month'].isin(month)]
    if processing_time:
        filtered_df = filtered_df[filtered_df['Processing Time'].isin(processing_time)]

    if filtered_df.empty:
        return [px.scatter() for _ in range(10)]  # Return empty graphs if filtered_df is empty

    return (
        px.bar(filtered_df, x='Region', y='Total Revenue', title="Total Sales Revenue by Region"),
        px.bar(filtered_df.groupby('Product')[['Unit Price', 'Total Cost']].mean().reset_index(), x='Product', y=['Unit Price', 'Total Cost'], title="Average Unit Price and Cost by Item Type"),
        px.bar(filtered_df.groupby('Country')['Profit'].sum().reset_index(), x='Country', y='Profit', title="Total Profit by Country"),
        px.bar(filtered_df.groupby('Sales Channel')['Order Priority'].count().reset_index(), x='Sales Channel', y='Order Priority', title="Order Priority Distribution by Sales Channel"),
        px.bar(filtered_df.groupby('Sales Channel')['Processing Time'].mean().reset_index(), x='Sales Channel', y='Processing Time', title="Average Order Processing Time by Sales Channel"),
        px.bar(filtered_df.groupby('Product')['Total Revenue'].sum().reset_index(), x='Product', y='Total Revenue', title="Total Sales by Item Type"),
        px.bar(filtered_df.groupby('Region')['Order Priority'].count().reset_index(), x='Region', y='Order Priority', title="Order Priority by Region"),
        px.scatter(filtered_df, x='Unit Price', y='Profit', title="Correlation between Unit Price and Total Profit"),
        px.line(filtered_df.groupby('Month')['Total Revenue'].sum().reset_index(), x='Month', y='Total Revenue', title="Seasonal Trends in Sales Data"),
        px.bar(filtered_df.groupby('Country')['Units Sold'].sum().reset_index(), x='Country', y='Units Sold', title="Units Sold by Country")
    )

# Callbacks for Advanced Insights Graphs
@app.callback(
    [Output('total-sales-revenue-by-country', 'figure'),
     Output('unit-price-distribution', 'figure'),
     Output('highest-average-unit-price', 'figure'),
     Output('total-cost-outliers', 'figure'),
     Output('total-profit-by-item-type', 'figure'),
     Output('average-processing-time-by-country', 'figure'),
     Output('highest-average-revenue-per-order', 'figure'),
     Output('units-sold-profit-correlation', 'figure'),
     Output('order-priority-by-item-type', 'figure'),
     Output('order-date-trends', 'figure')],
    [Input('dropdown1-advanced', 'value'),
     Input('dropdown2-advanced', 'value'),
     Input('dropdown3-advanced', 'value'),
     Input('dropdown4-advanced', 'value'),
     Input('dropdown5-advanced', 'value'),
     Input('dropdown6-advanced', 'value'),
     Input('dropdown7-advanced', 'value'),
     Input('dropdown8-advanced', 'value')]
)
def update_advanced_insights_graphs(region, year, country, product, order_priority, sales_channel, month, processing_time):
    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df['Region'].isin(region)]
    if year:
        filtered_df = filtered_df[filtered_df['Year'].isin(year)]
    if country:
        filtered_df = filtered_df[filtered_df['Country'].isin(country)]
    if product:
        filtered_df = filtered_df[filtered_df['Product'].isin(product)]
    if order_priority:
        filtered_df = filtered_df[filtered_df['Order Priority'].isin(order_priority)]
    if sales_channel:
        filtered_df = filtered_df[filtered_df['Sales Channel'].isin(sales_channel)]
    if month:
        filtered_df = filtered_df[filtered_df['Month'].isin(month)]
    if processing_time:
        filtered_df = filtered_df[filtered_df['Processing Time'].isin(processing_time)]

    if filtered_df.empty:
        return [px.scatter() for _ in range(10)]  # Return empty graphs if filtered_df is empty

    return (
        px.bar(filtered_df.groupby('Country')['Total Revenue'].sum().reset_index(), x='Country', y='Total Revenue', title="Total Sales Revenue by Country"),
        px.histogram(filtered_df, x='Unit Price', title="Unit Price Distribution by Item Type"),
        px.bar(filtered_df.groupby('Sales Channel')['Unit Price'].mean().reset_index(), x='Sales Channel', y='Unit Price', title="Highest Average Unit Price by Sales Channel"),
        px.box(filtered_df, y='Total Cost', title="Total Cost Outliers"),
        px.bar(filtered_df.groupby('Product')['Profit'].sum().reset_index(), x='Product', y='Profit', title="Total Profit by Item Type"),
        px.bar(filtered_df.groupby('Country')['Processing Time'].mean().reset_index(), x='Country', y='Processing Time', title="Average Order Processing Time by Country"),
        px.bar(filtered_df.groupby('Region')['Total Revenue'].mean().reset_index(), x='Region', y='Total Revenue', title="Highest Average Total Revenue per Order by Region"),
        px.scatter(filtered_df, x='Units Sold', y='Profit', title="Correlation between Units Sold and Total Profit"),
        px.bar(filtered_df.groupby('Product')['Order Priority'].count().reset_index(), x='Product', y='Order Priority', title="Order Priority by Item Type"),
        px.line(filtered_df.groupby('Month')['Total Revenue'].sum().reset_index(), x='Month', y='Total Revenue', title="Order Date Trends")
    )

# Callbacks for Data Analytics Graphs
@app.callback(
    [Output('analytics-graph1', 'figure'),
     Output('analytics-graph2', 'figure'),
     Output('analytics-graph3', 'figure'),
     Output('analytics-graph4', 'figure'),
     Output('analytics-graph5', 'figure'),
     Output('analytics-graph6', 'figure'),
     Output('analytics-graph7', 'figure'),
     Output('analytics-graph8', 'figure'),
     Output('analytics-graph9', 'figure'),
     Output('analytics-graph10', 'figure')],
    [Input('dropdown1-analytics', 'value'),
     Input('dropdown2-analytics', 'value'),
     Input('dropdown3-analytics', 'value'),
     Input('dropdown4-analytics', 'value'),
     Input('dropdown5-analytics', 'value'),
     Input('dropdown6-analytics', 'value'),
     Input('dropdown7-analytics', 'value'),
     Input('dropdown8-analytics', 'value')]
)
def update_data_analytics_graphs(region, year, country, product, order_priority, sales_channel, month, processing_time):
    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df['Region'].isin(region)]
    if year:
        filtered_df = filtered_df[filtered_df['Year'].isin(year)]
    if country:
        filtered_df = filtered_df[filtered_df['Country'].isin(country)]
    if product:
        filtered_df = filtered_df[filtered_df['Product'].isin(product)]
    if order_priority:
        filtered_df = filtered_df[filtered_df['Order Priority'].isin(order_priority)]
    if sales_channel:
        filtered_df = filtered_df[filtered_df['Sales Channel'].isin(sales_channel)]
    if month:
        filtered_df = filtered_df[filtered_df['Month'].isin(month)]
    if processing_time:
        filtered_df = filtered_df[filtered_df['Processing Time'].isin(processing_time)]

    if filtered_df.empty:
        return [px.scatter() for _ in range(10)]  # Return empty graphs if filtered_df is empty

    return (
        px.line(filtered_df, x='Order Date', y='Total Revenue', title="Sales Trend"),
        px.bar(filtered_df, x='Region', y='Total Revenue', title="Sales by Region"),
        px.bar(filtered_df, x='Country', y='Total Revenue', title="Total Sales by Country"),
        px.histogram(filtered_df, x='Unit Price', title="Unit Price Distribution"),
        px.scatter(filtered_df, x='Units Sold', y='Profit', title="Units Sold vs. Profit"),
        px.box(filtered_df, y='Total Cost', title="Total Cost Outliers"),
        px.bar(filtered_df, x='Region', y='Total Revenue', title="Highest Revenue Region"),
        px.bar(filtered_df, x='Order Priority', y='Units Sold', title="Order Priority vs. Items Sold"),
        px.line(filtered_df, x='Order Date', y='Units Sold', title="Order Date Trends"),
        px.pie(filtered_df, names='Product', values='Units Sold', title="Units Sold by Product")
    )

if __name__ == '__main__':
    app.run_server(debug=True)