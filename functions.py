import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
from arch import arch_model
from dotenv import load_dotenv
import os
import io
import boto3

load_dotenv()
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
KEY = os.getenv("aws_access_key_id")
SECRET = os.getenv("aws_secret_access_key")
s3 = boto3.client('s3',aws_access_key_id=KEY, aws_secret_access_key=SECRET)

def simulate_prices(prices, n_days, n_simulations=1000):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    last_price = prices.iloc[-1]
    dt = 1  # 1 day step

    # Monte Carlo simulation matrix
    simulations = np.zeros((n_days, n_simulations))

    for i in range(n_simulations):
        prices_path = [last_price]
        for _ in range(n_days):
            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.random.normal() * np.sqrt(dt)
            price = prices_path[-1] * np.exp(drift + shock)
            prices_path.append(price)
        simulations[:, i] = prices_path[1:]  # exclude starting price

    return simulations

def plot_simulation(df, simulations, input_data):
    mean_price = simulations.mean(axis=1).tolist()
    percentil_botton = input_data.percentil
    percentil_upper = 100-input_data.percentil
    percentile_5 = np.percentile(simulations, percentil_botton, axis=1).tolist()
    percentile_95 = np.percentile(simulations, percentil_upper, axis=1).tolist()

    last_date = df["Date"].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=input_data.n_days)

    df_last_year = df[df["Date"] > (df["Date"].max() - timedelta(days=365))]
    fig = go.Figure()

    # Línea azul: datos históricos del último año
    fig.add_trace(go.Scatter(x=df_last_year["Date"], y=df_last_year["Close"], mode='lines', name='Histórico', line=dict(color='blue')))

    # Línea roja: media simulada
    fig.add_trace(go.Scatter(x=forecast_dates, y=mean_price, mode='lines', name='Simulación media', line=dict(color='red')))

    # Percentil 5 (límite inferior)
    fig.add_trace(go.Scatter(x=forecast_dates, y=percentile_5, mode='lines', name=f'Percentil {percentil_botton}', line=dict(color='red', dash='dot')))

    # Percentil 95 (límite superior)
    fig.add_trace(go.Scatter(x=forecast_dates, y=percentile_95, mode='lines', name=f'Percentil {percentil_upper}', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title=f"Simulación Monte Carlo: {input_data.commodity.title()}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_white"
    )
    html_filename = f"html_result/{input_data.commodity.title()}_simulation.html"
    fig.write_html(html_filename)
    graph_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    return graph_json
    
def plot_volatility_forecast(garch_vol, forecast, req):
    forecast_dates = pd.date_range(start=garch_vol.index[-1] + timedelta(days=1), periods=req.n_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=garch_vol.index, y=garch_vol, name="GARCH Volatility", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Forecast (NN)", line=dict(color="orange", dash="dash")))

    fig.update_layout(title=f"Predicción de Volatilidad - {req.commodity.title()}", xaxis_title="Fecha", yaxis_title="Volatilidad")

    html_path = f"html_result/{req.commodity}_vol_forecast.html"
    fig.write_html(html_path)

    return {
        "message": "Forecast generado",
        "forecast_html": html_path,
        "plot": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    }

def estimated_volatility_garch(path, p=1,q=1):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    returns = 100 * df["Close"].pct_change().dropna()
    returns.index = df["Date"].iloc[1:]

    garch_model = arch_model(returns, vol='Garch', p=p, q=q)
    garch_result = garch_model.fit(disp='off')
    garch_vol = garch_result.conditional_volatility

    return garch_vol

def upload_s3(df, file_name: str):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=BUCKET_NAME, Key=file_name, Body=csv_buffer.getvalue())