# Commodity Prices-Volatility API
Una API construida con FastAPI que permite consultar precios históricos de commodities, ejecutar simulaciones de Monte Carlo para proyección de precios y predecir la volatilidad futura de los activos mediante modelos de Machine Learning.

## Funcionalidades principales
🔹 Consulta de precios históricos  
🔹 Simulación de escenarios de precios mediante Monte Carlo  
🔹 Calibración de modelos de predicción de volatilidad (GARCH + LSTM)  
🔹 Predicción de volatilidad futura  

## Como ejecutar
### Clonar Repositorio
```bash
 git clone https://github.com/crcordova/comodity-prices
 cd comodity-prices
```

### Instalar dependencias
```bash
 pip install -r requirements.txt
```

### Configurar acceso bucket S3 AWS
copia el file `.env`
```bash
cp .env.example .env
```  
edita y configira tus claves

### Configuracion de activos a rastrear
editar archivo `commodity_tickers.json` 
```bash
{
    "Activo": "Ticket YahooFinance"
    "Copper": "HG=F",
    "Dolar": "CLP=X"
}
```

### Iniciar API
```bash
 uvicorn main:app --reload
```

La API estará disponible en  `http://localhost:8000/docs`

## Tecnologías y librerías

 - FastAPI – Framework backend rápido y moderno

 - yfinance / pandas – Obtención y procesamiento de datos de mercado

 - arch / statsmodels – Modelado GARCH para volatilidad

 - TensorFlow / Keras – Modelo LSTM para predicción temporal

 - NumPy / Plotly – Simulación y visualización