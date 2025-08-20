from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import json
import numpy as np
import os
import shutil
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from keras.losses import MeanSquaredError
import joblib
from typing import List

from models import SimulationInput, CalibrationRequest, ForecastRequest, ForecastInput
from functions import (
    simulate_prices, plot_simulation, plot_volatility_forecast, 
    estimated_volatility_garch, upload_s3, load_and_generate_features, 
    create_targets, train_random_forest_range, evaluate_prediction,
    predict_future_range
)

app = FastAPI(title="API Financiera Simulación de precios", version="1.0.0")
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # permite frontend
    allow_credentials=True,
    allow_methods=["*"],     # permite todos los métodos, incluido OPTIONS
    allow_headers=["*"],
)

with open("commodity_tickers.json") as f:
    ticker_map = json.load(f)

@app.get("/commodities", response_model=List[str])
def get_commodities():

    try:
        commodities = list(ticker_map.keys())
        return commodities
    except Exception as e:
        raise RuntimeError(f"Error leyendo el archivo: {e}")

@app.get("/get_prices/")
async def get_prices(commodity: str):
    '''
    Descarga precios historicos del commodity, este debe estar listado en archivo .json
    '''
    if commodity == "Zinc":
        return{"message": f"{commodity} commodity should be updated manually please contact support"}
    
    ticker = ticker_map.get(commodity)
    if not ticker:
        raise HTTPException(status_code=404, detail="Commodity not found")
    
    data = yf.download(ticker, period="5y")[['Close']]
    data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)

    filename = f"historical_data/{commodity}_prices.csv"
    data.to_csv(filename, index=False)
    upload_s3(data,f"prices/{commodity}_prices.csv")
    
    return {"message": f"Data saved to {filename}"}

@app.post("/simulate_price")
def simulate(input_data: SimulationInput):
    '''
    Con los precios historicos del commodity ya descargados, estos son usados para generar una simula a "n" días
    '''
    filename = f"historical_data/{input_data.commodity}_prices.csv"
    #TODO leer desde S3
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    if "Close" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'price' column.")
    
    simulations = simulate_prices(df["Close"], input_data.n_days)

    # Extract summary statistics for each day
    mean_price = simulations.mean(axis=1).tolist()
    percentile_5 = np.percentile(simulations, input_data.percentil, axis=1).tolist()
    percentile_95 = np.percentile(simulations, 100-input_data.percentil, axis=1).tolist()

    df_res = pd.DataFrame([[input_data.commodity, input_data.n_days, mean_price[-1],percentile_95[-1], percentile_5[-1]]],columns=['Commodity','days','mean','upper','lower'])
    df_res.to_csv(f'MC_result/{input_data.commodity}_sim.csv', index=False)
    upload_s3(df_res,f"{input_data.commodity}_sim.csv")

    graph_json = plot_simulation(df, simulations, input_data, save_local=False)

    return {
        "plot": graph_json,
        "meta": {
            "last_price": df['Close'].iloc[-1],
            "days": input_data.n_days,
            'upper_price': percentile_95[-1],
            'lower_price': percentile_5[-1],
        }
    }

@app.post("/calibrate-volatility")
def calibrate_volatility(req: CalibrationRequest):
    '''
    Con los precios historicos se calibra un modelo de predicción de volatilidad utilizando GARCH y Redes Neuronales
    '''
    path = f"historical_data/{req.commodity}_prices.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CSV not found")

    garch_vol = estimated_volatility_garch(path, p=1,q=1)

    X, y = [], []
    for i in range(req.look_back, len(garch_vol)):
        X.append(garch_vol[i - req.look_back:i])
        y.append(garch_vol[i])
    X, y = np.array(X), np.array(y)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(req.look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y_scaled, epochs=req.epochs, batch_size=32, verbose=0)

    os.makedirs("ML_models", exist_ok=True)
    model.save(f"ML_models/{req.commodity}_lstm.h5")
    joblib.dump(scaler_X, f"ML_models/{req.commodity}_scaler_X.pkl")
    joblib.dump(scaler_y, f"ML_models/{req.commodity}_scaler_y.pkl")

    return {"message": "Modelo calibrado y guardado con éxito"}


@app.post("/forecast-volatility")
def forecast_volatility(req: ForecastRequest):
    path = f"historical_data/{req.commodity}_prices.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CSV not found")

    garch_vol = estimated_volatility_garch(path, p=1,q=1)

    look_back = 10
    last_sequence = garch_vol[-look_back:].values

    model = load_model(f"ML_models/{req.commodity}_lstm.h5", compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())
    scaler_X = joblib.load(f"ML_models/{req.commodity}_scaler_X.pkl")
    scaler_y = joblib.load(f"ML_models/{req.commodity}_scaler_y.pkl")

    forecast = []
    for _ in range(req.n_days):
        input_seq = scaler_X.transform(last_sequence.reshape(1, -1)).reshape((1, look_back, 1))
        pred_scaled = model.predict(input_seq, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        forecast.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)

    response = plot_volatility_forecast(garch_vol, forecast, req)
    return response

@app.post("/forecast-price")
def forecast_price(req: ForecastInput):
    '''Forecast commodity prices using RandomForest and some financial indicators'''
    filename = f"historical_data/{req.commodity}_prices.csv"
    #TODO leer desde S3
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    
    df = load_and_generate_features(filename)
    df, df_fin_test = create_targets(df, req.n_days)
    model, X_test, y_test, y_pred = train_random_forest_range(df, n_days=5)
    response = evaluate_prediction(df, model, X_test, y_test, n_eval=100)
    response['predictions'] = predict_future_range(df_fin_test, model, n_days=5)
    return response

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Validar que sea CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .csv")

    file_path = os.path.join('historical_data', "Zinc_prices.csv")

    # Guardar archivo en la carpeta historical_data
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={
        "message": "Archivo guardado correctamente",
        "file_path": file_path
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)