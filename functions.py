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
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def plot_simulation(df, simulations, input_data, save_local=False):
    mean_price = simulations.mean(axis=1).tolist()
    percentil_botton = input_data.percentil
    percentil_upper = 100-input_data.percentil
    percentile_5 = np.percentile(simulations, percentil_botton, axis=1).tolist()
    percentile_95 = np.percentile(simulations, percentil_upper, axis=1).tolist()

    last_date = df["Date"].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=input_data.n_days)

    six_months_ago = df["Date"].max() - timedelta(days=180)
    # df_last = df[df["Date"] > (df["Date"].max() - timedelta(days=365))]
    df_last = df[df["Date"] >= six_months_ago]

    fig = go.Figure()

    # Línea azul: histórico últimos 6 meses
    fig.add_trace(go.Scatter(
        x=df_last["Date"],
        y=df_last["Close"],
        mode='lines',
        name='Histórico',
        line=dict(color='blue')
    ))

    # Línea roja: media simulada
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=mean_price,
        mode='lines',
        name='Simulación media',
        line=dict(color='red')
    ))

    # Banda sombreada entre percentil bajo y alto
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),  # unir superior + inferior en un polígono
        y=percentile_95 + percentile_5[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',  # rojo claro semitransparente
        line=dict(color='rgba(255,255,255,0)'),  # sin borde
        name=f'Rango percentil {percentil_botton}-{percentil_upper}',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Simulación Monte Carlo: {input_data.commodity.title()}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_white"
    )

    if save_local:
        html_filename = f"html_result/{input_data.commodity.title()}_simulation.html"
        fig.write_html(html_filename)

    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def plot_volatility_forecast(garch_vol, forecast, req, save_local=False):

    history_days = req.n_days + 60  # usamos el mismo número de días que el forecast
    garch_vol_filtered = garch_vol[garch_vol.index >= garch_vol.index[-history_days]]
    forecast_dates = pd.date_range(start=garch_vol.index[-1] + timedelta(days=1), periods=req.n_days)
    save_volatility_to_s3(forecast_dates, forecast, req.commodity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=garch_vol_filtered.index, y=garch_vol_filtered, name="GARCH Volatility", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Forecast (NN)", line=dict(color="orange", dash="dash")))

    fig.update_layout(title=f"Predicción de Volatilidad - {req.commodity.title()}", xaxis_title="Fecha", yaxis_title="Volatilidad")

    if save_local==True:
        html_path = f"html_result/{req.commodity}_vol_forecast.html"
        fig.write_html(html_path)

    return {
        "message": "Forecast generado",
        # "forecast_html": html_path,
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


def load_and_generate_features(csv_file):
    """
    Lee archivo CSV con columnas Date y Close, y genera features técnicos.
    Opcionalmente genera la columna objetivo 'target' con horizonte n_target días.
    """
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calcular retorno logarítmico
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Medias móviles
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatilidad (desviación estándar de retornos)
    df['volatility_10'] = df['log_return'].rolling(window=10).std(ddof=0)
    df['volatility_20'] = df['log_return'].rolling(window=20).std(ddof=0)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Eliminar filas con NaN
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_targets(df, n_days):
    """
    Recibe un DataFrame con al menos columnas Date, Close, log_return y volatility (o volatility_X).
    Agrega columnas con el retorno futuro a n días y bandas superior/inferior usando la volatilidad actual.
    """
    df = df.copy()
    
    # Retorno futuro acumulado a n días
    future_close = df['Close'].shift(-n_days)
    df[f'ret_future_{n_days}'] = (future_close - df['Close']) / df['Close']
    
    # Usamos la volatilidad actual; si tienes varias puedes elegir la que te interese:
    # Ej: 'volatility_20' o 'volatility_10'. Aquí uso volatility_10 como ejemplo:
    if 'volatility_10' in df.columns:
        vol_col = 'volatility_10'
    else:
        # fallback genérico
        vol_col = [col for col in df.columns if 'volatility' in col][0]
    
    df['target_upper'] = df[f'ret_future_{n_days}'] + df[vol_col]*2
    df['target_lower'] = df[f'ret_future_{n_days}'] - df[vol_col]*2
    
    # Opcional: eliminar filas con NaN producidas al final
    df_final_test = df.tail(1)[['Date','Close','sma_20', 'sma_50', 'volatility_10', 'volatility_20', 'macd', 'signal']]
    df = df.dropna(subset=[f'ret_future_{n_days}', 'target_upper', 'target_lower'])

    return df, df_final_test

def train_random_forest_range(df, n_days=5):
    """
    Entrena dos RandomForest (MultiOutput) para predecir target_upper y target_lower.
    Retorna el modelo entrenado y conjunto X_test, y_test, y predicciones.
    """

    # Definir features y targets
    feature_cols = ['sma_20', 'sma_50', 'volatility_10', 'volatility_20', 'macd', 'signal']
    X = df[feature_cols]
    y = df[['target_upper', 'target_lower']]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # sin shuffle porque es serie temporal
    )

    # Modelo RandomForest multisalida
    base_model = RandomForestRegressor(n_estimators=200, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

def evaluate_prediction(df, model, X_test, y_test, n_eval = 100):
    X_eval = X_test.tail(n_eval)
    y_eval = y_test.tail(n_eval)

    # Predicciones
    y_pred_eval = model.predict(X_eval)

    # Convertimos a DataFrame para juntar todo
    eval_df = X_eval.copy()
    eval_df['ret_future_5'] = df[f'ret_future_5'][-len(X_test):].tail(n_eval).values
    eval_df['pred_upper'] = y_pred_eval[:, 0]
    eval_df['pred_lower'] = y_pred_eval[:, 1]
    eval_df['target_upper_real'] = y_eval.iloc[:, 0].values
    eval_df['target_lower_real'] = y_eval.iloc[:, 1].values

    # Ver si el retorno real cae dentro del rango predicho
    eval_df['is_covered'] = eval_df.apply(lambda row: row['pred_lower'] <= row['ret_future_5'] <= row['pred_upper'], axis=1)

    coverage_rate = eval_df['is_covered'].mean()

    # Necesitamos extraer la columna Date correspondiente
    # Suponiendo que la porción de df_targets usada como test es secuencial (sin shuffle)
    dates_test = df['Date'][-len(X_test):].tail(n_eval).values

    # Construimos DataFrame para evaluación
    eval_df = pd.DataFrame({
        'Date': dates_test,
        'ret_future_5': df['ret_future_5'][-len(X_test):].tail(n_eval).values,
        'pred_upper': y_pred_eval[:, 0],
        'pred_lower': y_pred_eval[:, 1]
    })

    eval_df['is_covered'] = eval_df.apply(lambda row: row['pred_lower'] <= row['ret_future_5'] <= row['pred_upper'], axis=1)
    coverage_rate = eval_df['is_covered'].mean()

    # Plot con fechas
    fig = go.Figure()

    # Franja inferior/superior
    fig.add_trace(go.Scatter(
        x=eval_df['Date'], y=eval_df['pred_upper'],
        mode='lines', name='Predicted Upper', line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=eval_df['Date'], y=eval_df['pred_lower'],
        mode='lines', name='Predicted Lower', line=dict(dash='dash')
    ))

    # Banda sombreada
    fig.add_trace(go.Scatter(
        x=pd.concat([eval_df['Date'], eval_df['Date'][::-1]]),
        y=pd.concat([eval_df['pred_upper'], eval_df['pred_lower'][::-1]]),
        fill='toself', fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=False
    ))

    # Línea real
    fig.add_trace(go.Scatter(
        x=eval_df['Date'], y=eval_df['ret_future_5'],
        mode='lines+markers', name='Real Return 5d'
    ))

    fig.update_layout(
        title=f"5-Day Return vs Predicted Range - Latest {n_eval} days",
        xaxis_title="Date",
        yaxis_title="Return (proportion)",
        legend=dict(x=0.01, y=0.99)
    )

    return {'coverage_rate': coverage_rate, 'plot': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}

def predict_future_range(df, model, n_days=5,):
    '''Predice con el ultimo dato del dataframe original'''
    features = ['sma_20', 'sma_50', 'volatility_10', 'volatility_20', 'macd', 'signal']
    pred = model.predict(df[features])
    pred_upper, pred_lower = pred[0]
    
    # Convertir retornos en precios futuros
    last_close = df['Close'].iloc[-1]
    price_upper = last_close * (1 + pred_upper)
    price_lower = last_close * (1 + pred_lower)
    
    result = {
        "date": df['Date'].iloc[-1],
        "last_close": last_close,
        f"pred_upper_ret": pred_upper,
        f"pred_lower_ret": pred_lower,
        f"pred_upper_price": price_upper,
        f"pred_lower_price": price_lower
    }
    
    return result

def upload_s3(df, file_name: str):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=BUCKET_NAME, Key=file_name, Body=csv_buffer.getvalue())

def save_forecast_to_s3(response: dict, commodity_name: str):
    """
    Convierte el response en CSV y lo guarda en S3.
    """
    coverage_rate = response.get("coverage_rate")
    predictions = response.get("predictions")
    if isinstance(predictions, dict):
        predictions = [predictions]

    # Convertir Timestamps y np.float64 a tipos nativos
    for pred in predictions:
        for k, v in pred.items():
            if isinstance(v, pd.Timestamp):
                pred[k] = v.strftime("%Y-%m-%d")
            elif hasattr(v, "item"):  # np.float64 -> float
                pred[k] = float(v)

    df = pd.DataFrame(predictions)

    # Agregar coverage_rate a todas las filas
    df["coverage_rate"] = coverage_rate

    # Convertir DataFrame a CSV (en memoria)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Nombre del archivo
    file_name = f"{commodity_name}_forecast_price.csv"

    # Subir a S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=file_name,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )

def save_volatility_to_s3(forecast_dates, forecast, commodity_name):
    """
    Guarda la volatilidad proyectada en un CSV en S3.
    """
    # Asegurarnos que forecast_dates y forecast tengan la misma longitud
    if len(forecast_dates) != len(forecast):
        raise ValueError("forecast_dates y forecast deben tener la misma longitud")

    # Crear lista de diccionarios
    data = []
    for date, vol in zip(forecast_dates, forecast):
        # Convertir np.float64 a float y Timestamp a string
        if hasattr(vol, "item"):
            vol = float(vol)
        if isinstance(date, pd.Timestamp):
            date = date.strftime("%Y-%m-%d")
        data.append({"date": date, "forecast_volatility": vol})

    # Crear DataFrame
    df = pd.DataFrame(data)

    # CSV en memoria
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Subir a S3
    file_name = f"{commodity_name}_forecast_volatility.csv"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=file_name,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )

def read_historical_prices_s3(file_s3 : str):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_s3)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data), parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df