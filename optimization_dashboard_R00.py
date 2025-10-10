import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import date
import base64
from io import BytesIO
import plotly.express as px
import pandas as pd
import numpy as np
import re

# Import the corrected analysis scripts as modules
import C00_analysis_R00
import C01_analysis_R00
import C02_analysis_R00
import C03_analysis_R00

# --- 1. Constants and Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# CONFIGURABLE PATH FOR THE GC ANALYSIS SHEET
# !!! PLEASE UPDATE THIS PATH TO THE CORRECT LOCATION OF YOUR GC DATA FILE !!!
GC_DATA_FILE_PATH = r"E:\GENERATING_DATA\WFO_Plant_GC_Report-25-26.csv"

# Define the data storage for the plots
ANALYSIS_DATA_STORE = dcc.Store(id='analysis-data-store', data={})

# --- 2. Layout (User Interface) ---
app.layout = dbc.Container([
    ANALYSIS_DATA_STORE,
    dbc.Row(dbc.Col(html.H1("Distillation Column Performance Analysis", className="text-center my-4"))),
    
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='column-selector',
            options=[
                {'label': 'C-00 (Crude Tar Distillation)', 'value': 'C-00'},
                {'label': 'C-01 (Anthracene Oil Recovery)', 'value': 'C-01'},
                {'label': 'C-02 (Benzene Column)', 'value': 'C-02'},
                {'label': 'C-03 (Naphthalene Column)', 'value': 'C-03'}
            ],
            placeholder="Select a Distillation Column...",
            className="mb-3"
        )),
        dbc.Col(dcc.DatePickerRange(
            id='date-picker-range',
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            display_format='YYYY-MM-DD',
            className="mb-3"
        )),
        dbc.Col(dbc.Button("Run Analysis", id='run-button', color="primary", className="mb-3")),
    ]),

    dbc.Row(dbc.Col(html.Div(id='analysis-status', className="text-center my-3"))),

    dbc.Row(dbc.Col(html.Div(id='analysis-results-output', className="my-3"))),

    dbc.Row(dbc.Col(html.Div(id='analysis-plots-output', className="my-3"))),

    dbc.Row(dbc.Col(dbc.Button("Generate Word Report", id="download-button", color="success", className="my-3", style={'display': 'none'}))),
    dcc.Download(id="download-word-report")

], fluid=True)

# --- 3. Helper Functions ---
def get_module(column_name):
    """Dynamically returns the correct analysis module based on the column name."""
    modules = {
        'C-00': C00_analysis_R00,
        'C-01': C01_analysis_R00,
        'C-02': C02_analysis_R00,
        'C-03': C03_analysis_R00,
    }
    return modules.get(column_name)

def generate_report_content(analysis_data):
    """Generates the Word report using the data from the store."""
    if not analysis_data or not analysis_data.get('column'):
        return None

    column_name = analysis_data['column']
    module = get_module(column_name)
    if not module:
        return None

    analysis_results = analysis_data['results']
    plots_data = analysis_data['plots']
    plot_flags = analysis_data['plot_flags']

    # Convert base64 plot data back to BytesIO objects
    plot_buffers = {
        plot_id: BytesIO(base64.b64decode(b64_str)) if b64_str else None
        for plot_id, b64_str in plots_data.items()
    }

    start_date = analysis_data.get('start_date')
    end_date = analysis_data.get('end_date')
    
    # We now pass the original analysis DataFrame to the report function
    df = pd.DataFrame(analysis_data.get('dataframe'))

    doc_buffer = module.generate_word_report(analysis_results, df, {}, start_date, end_date, plot_buffers, plot_flags)

    return doc_buffer

# --- 4. Callbacks (Application Logic) ---

@app.callback(
    Output('analysis-data-store', 'data'),
    Output('analysis-status', 'children'),
    Output('download-button', 'style'),
    Input('run-button', 'n_clicks'),
    State('column-selector', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def run_analysis_callback(n_clicks, column_name, start_date, end_date):
    """Runs the analysis for the selected column and stores the results."""
    
    if not n_clicks or not column_name or not start_date or not end_date:
        return {}, "Please select a column and a date range.", {'display': 'none'}

    module = get_module(column_name)
    if not module:
        return {}, f"Error: No analysis module found for {column_name}.", {'display': 'none'}

    status_message = f"Running analysis for {column_name} from {start_date} to {end_date}..."
    
    try:
        # Pass the configurable GC file path to the analysis function
        analysis_results, df, outliers, plot_buffers, plot_flags = module.run_cXX_analysis(start_date, end_date, gc_file_path=GC_DATA_FILE_PATH)
    except Exception as e:
        return {}, f"Error during analysis: {e}", {'display': 'none'}

    if not analysis_results:
        return {}, "Analysis failed or no data found for the selected period.", {'display': 'none'}

    # Convert plot buffers to base64 strings for storage
    base64_plots = {
        plot_id: base64.b64encode(buf.getvalue()).decode('utf-8') if buf else None
        for plot_id, buf in plot_buffers.items()
    }
    
    # Store the dates and a serializable version of the DataFrame for the report
    analysis_data = {
        'column': column_name,
        'results': analysis_results,
        'plots': base64_plots,
        'plot_flags': plot_flags,
        'start_date': start_date,
        'end_date': end_date,
        'dataframe': df.to_dict('records') # Convert DataFrame to a serializable dictionary
    }

    status_message = "Analysis complete. Results displayed below."
    return analysis_data, status_message, {'display': 'block'}


@app.callback(
    Output('analysis-results-output', 'children'),
    Output('analysis-plots-output', 'children'),
    Input('analysis-data-store', 'data')
)
def update_dashboard_content(analysis_data):
    """Updates the dashboard with the stored analysis results and plots."""
    if not analysis_data:
        return html.Div(), html.Div()

    results = analysis_data['results']
    plots = analysis_data['plots']
    column = analysis_data['column']
    
    # Generate the KPI table
    kpi_rows = []
    # Use a specific order for the KPIs
    kpi_order = ['Average Feed Flow', 'Overall Material Balance Error (%)', 'Average Top Product Flow', 'Average Bottom Product Flow', 'Average Naphthalene Loss (%)', 'Average Top Product Purity', 'Average Bottom Product Purity', 'Average Differential Pressure', 'Average Reboiler Heat Duty', 'Average Condenser Heat Duty']
    
    for key in kpi_order:
        if key in results:
            value = results[key]
            tag_match = re.search(r'\((.*?)\)', key)
            tag = tag_match.group(1) if tag_match else key
            unit = C00_analysis_R00.TAG_UNITS.get(tag, '') # Use one of the modules to get units
            if isinstance(value, (float, int)):
                kpi_rows.append(html.Tr([
                    html.Td(key),
                    html.Td(f"{value:.2f} {unit}"),
                ]))
            else:
                kpi_rows.append(html.Tr([
                    html.Td(key),
                    html.Td(value),
                ]))

    kpi_table = dbc.Table(html.Tbody(kpi_rows), bordered=True, striped=True, hover=True, responsive=True)
    
    results_div = html.Div([
        html.H3(f"Key Performance Indicators for {column}", className="my-3"),
        kpi_table
    ])

    # Generate the plots
    plot_divs = []
    for plot_id, b64_str in plots.items():
        if b64_str:
            plot_divs.append(
                html.Div([
                    html.H4(plot_id.replace('_', ' ').title(), className="mt-4"),
                    html.Img(src=f"data:image/png;base64,{b64_str}", style={'width': '100%', 'height': 'auto'})
                ])
            )
    
    plots_div = html.Div(plot_divs)
    
    return results_div, plots_div

@app.callback(
    Output("download-word-report", "data"),
    Input("download-button", "n_clicks"),
    State('analysis-data-store', 'data'),
    prevent_initial_call=True
)
def download_report(n_clicks, analysis_data):
    """Triggers the Word report download."""
    if n_clicks is None or not analysis_data:
        return dash.no_update

    doc_buffer = generate_report_content(analysis_data)
    
    if doc_buffer:
        return dcc.send_bytes(doc_buffer.getvalue(), f"{analysis_data['column']}_Performance_Report.docx")
    
    return dash.no_update

# --- 5. Run the Application ---
if __name__ == '__main__':
    app.run(debug=True)