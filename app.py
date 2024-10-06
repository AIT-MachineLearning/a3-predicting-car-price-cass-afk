import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import dash_bootstrap_components as dbc  # For styling

# Load your machine learning model
pred_model = joblib.load("model.pkl")
scale_model = joblib.load("a3_scaler.pkl")

# Initialize the Dash app with Bootstrap for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app with a beautiful interface
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Car Price Prediction", className="text-center text-primary mb-4"),
                width=12,
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Engine (cc)"),
                        dbc.Input(id="engine-data", type="number", placeholder="Enter Engine size"),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Max Power (bhp)"),
                        dbc.Input(id="max-power-data", type="number", placeholder="Enter Max Power"),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Mileage (kmpl)"),
                        dbc.Input(id="mil-data", type="number", placeholder="Enter Mileage"),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Kilometers Driven"),
                        dbc.Input(id="km-data", type="number", placeholder="Enter KMs Driven"),
                    ],
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Predict Price", id="submit-val", color="primary", className="mt-4", n_clicks=0),
                width=12,
                className="text-center",
            )
        ),
        dbc.Row(
            dbc.Col(
                html.H3(id="output-prediction", className="text-center text-success mt-4"),
                width=12,
            )
        ),
    ],
    fluid=True,
)

# Define the callback to update the prediction
@app.callback(
    Output("output-prediction", "children"),
    [Input("submit-val", "n_clicks")],
    [State("engine-data", "value"),
     State("max-power-data", "value"),
     State("mil-data", "value"),
     State("km-data", "value")]
)
def update_output(n_clicks, value1, value2, value3, value4):
    if n_clicks > 0:
        # Check for None inputs and replace them with default values
        if value1 is None:
            value1 = 1248.0
        if value2 is None:
            value2 = 90.0
        if value3 is None:
            value3 = 18.80
        if value4 is None:
            value4 = 100000

        # Prepare the input data
        input_data = pd.DataFrame([[value1, value2, value3, value4]],
                                  columns=['engine', 'max_power', 'mileage', 'km_driven'])
        scaled_data = scale_model.transform(input_data)

        # Get the prediction from the model
        prediction = pred_model.predict(scaled_data)

        # Return the formatted prediction
        if prediction[0] == 0:
            return f"Predicted Price Bucket:  {(prediction[0])} in range 29999.0, 272499.25"

        elif prediction[0] == 1:
            return f"Predicted Price Bucket:  {(prediction[0])} in range 272499.25, 514999.5"

        elif prediction[0]==2:
            return f"Predicted Price Bucket:  {(prediction[0])} in range  514999.5, 757499.75"

        else:
            return f"Predicted Price Bucket:  {(prediction[0])} in range 757499.75, 1000000.0"



    return "Waiting for input"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=7070)
