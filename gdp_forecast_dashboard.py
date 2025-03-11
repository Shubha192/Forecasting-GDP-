import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


st.set_page_config(
    page_title="GDP Forecast Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {

    background-color: #E8EEF6 !important;
}

.main-title {
    background-color: #2C3E50;
    color: white !important;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 36px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.section-header {
    background-color: #34495E;
    color: white !important;
    padding: 15px;
    border-radius: 8px;
    margin: 20px 0;
    font-size: 24px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 10px 0;
    border-left: 5px solid #3498DB;
}

.plot-container {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
}

.dataframe {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

def load_data():
    file_path = 'C:/Users/Rushikesh/OneDrive/Desktop/mlt/Global Economy Indicators.csv'
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    gdp_data = data[['Year', 'Gross Domestic Product (GDP)']].dropna()
    gdp_data = gdp_data.groupby('Year')['Gross Domestic Product (GDP)'].mean()
    gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y')
    gdp_data = gdp_data[gdp_data.index >= '1970']
    gdp_data = gdp_data[gdp_data.index <= '2025']
    gdp_data.index = pd.date_range(start=gdp_data.index[0], periods=len(gdp_data), freq='Y')
    return gdp_data

def load_model():
    with open('gdp_arima_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def create_historical_analysis_plot(actual_data, historical_predictions):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data.index,
        y=actual_data,
        name='Actual GDP',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=actual_data.index,
        y=historical_predictions,
        name='Model Predictions',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Historical GDP vs Model Predictions (1970-2025)',
        xaxis_title='Year',
        yaxis_title='GDP',
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_forecast_plot(historical_data, model_predictions, forecast_data, forecast_dates):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data,
        name='Historical GDP',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=model_predictions,
        name='Historical Predictions',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['Predicted_GDP'],
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
        y=forecast_data['Upper_CI'].tolist() + forecast_data['Lower_CI'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title='GDP Forecast Analysis',
        xaxis_title='Year',
        yaxis_title='GDP',
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def main():
        st.markdown('<div class="main-title">GDP Forecast Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 20px; color: #2C3E50; margin-bottom: 30px;">Analysis and Predictions from 1970 to Future</div>', unsafe_allow_html=True)
    
    try:
               gdp_data = load_data()
        model = load_model()
        
   
        historical_predictions = np.exp(model.get_prediction(start=0).predicted_mean)
        forecast_steps = 10
        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = np.exp(forecast.predicted_mean)
        forecast_ci = np.exp(forecast.conf_int())
        
           last_date = gdp_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), 
                                     periods=forecast_steps, 
                                     freq='Y')
        
                forecast_df = pd.DataFrame({
            'Predicted_GDP': forecast_mean,
            'Lower_CI': forecast_ci.iloc[:, 0],
            'Upper_CI': forecast_ci.iloc[:, 1]
        }, index=forecast_dates)
        
               col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mean Squared Error", f"{mean_squared_error(gdp_data, historical_predictions):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("MAPE", f"{mean_absolute_percentage_error(gdp_data, historical_predictions)*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Data Points", len(gdp_data))
            st.markdown('</div>', unsafe_allow_html=True)
        
              st.markdown('<div class="section-header">Historical Analysis (1970-2025)</div>', unsafe_allow_html=True)
        historical_fig = create_historical_analysis_plot(gdp_data, historical_predictions)
        st.plotly_chart(historical_fig, use_container_width=True)
        
               st.markdown('<div class="section-header">Future Forecast</div>', unsafe_allow_html=True)
        forecast_fig = create_forecast_plot(gdp_data, historical_predictions, forecast_df, forecast_dates)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
              col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header" style="font-size: 20px;">Historical Data</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                'Actual_GDP': gdp_data,
                'Predicted_GDP': historical_predictions
            }))
        
        with col2:
            st.markdown('<div class="section-header" style="font-size: 20px;">Forecast Values</div>', unsafe_allow_html=True)
            st.dataframe(forecast_df)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure to run model.py first to generate the model file.")

if _name_ == "_main_":
    main()
