import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

file_path = 'C:/Users/Rushikesh/OneDrive/Desktop/mlt/Global Economy Indicators.csv'
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
gdp_data = data[['Year', 'Gross Domestic Product (GDP)']].dropna()
gdp_data = gdp_data.groupby('Year')['Gross Domestic Product (GDP)'].mean()

gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y')
gdp_data = gdp_data[gdp_data.index >= '1970']
gdp_data = gdp_data[gdp_data.index <= '2025']

gdp_data.index = pd.date_range(start=gdp_data.index[0], periods=len(gdp_data), freq='Y')

train_size = int(len(gdp_data) * 0.8)
train_data = gdp_data[:train_size]
test_data = gdp_data[train_size:]

gdp_log = np.log(gdp_data)
train_log = np.log(train_data)

best_aic = float('inf')
best_params = None
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(train_log, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, d, q)
            except:
                continue

print(f"Best ARIMA parameters: {best_params}")

final_model = ARIMA(gdp_log, order=best_params)
results = final_model.fit()
forecast_steps = 10
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = np.exp(forecast.predicted_mean)
forecast_ci = np.exp(forecast.conf_int())

historical_predictions = np.exp(results.get_prediction(start=0).predicted_mean)
last_date = gdp_data.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=forecast_steps, freq='Y')

with open('gdp_arima_model.pkl', 'wb') as f:
    pickle.dump(results, f)

mse = mean_squared_error(gdp_data, historical_predictions)
mape = mean_absolute_percentage_error(gdp_data, historical_predictions)
print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Percentage Error: {mape*100:.2f}%")

print("\nHistorical GDP Data (1970-2025):")
print(gdp_data)

print("\nForecast Results:")
forecast_df = pd.DataFrame({
    'Predicted_GDP': forecast_mean,
    'Lower_CI': forecast_ci.iloc[:, 0],
    'Upper_CI': forecast_ci.iloc[:, 1]
}, index=forecast_dates)
print(forecast_df)

plt.figure(figsize=(12, 6))
plt.plot(gdp_data.index, gdp_data, label='Actual Data', color='blue')
plt.plot(gdp_data.index, historical_predictions, label='Historical Predictions', color='green', linestyle='--')
plt.plot(forecast_dates, forecast_mean, color='r', label='Future Forecast')
plt.fill_between(forecast_dates, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='r', 
                 alpha=0.1)
plt.title('GDP Historical Data, Predictions, and Forecast')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.show()
