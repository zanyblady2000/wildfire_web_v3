import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

rfc = joblib.load('rfc_model.joblib')
scaler = joblib.load('scaler.joblib')

def Inputs():
  st.sidebar.header = ("Inputs For Prediction")

  temp = st.sidebar.slider = ('Temperature (C*)', 15, 35, 0)
  wind_speed = st.sidebar.slider = ('Wind_Speed (Km/h)', 0, 50, 0)
  humidity = st.sidebar.slider = ('Humidity (%)', 0, 100, 0)
  LDSR = st.sidebar.slider = ('Last Day Since Rain)', 0, 7, 0)

  st.sidebar.subheader = ("Location Inputs")
  lat = st.sidebar.slider('Latitude', 50, 59, 55)
  long = st.sidebar.slider('Longitude', -124, -113, -118)

  data = {
        'temp': temp, 
        'humidity': humidity, 
        'windspeed': windspeed,
        'Last Day Since Rain': LDSR,
        'lat': lat, 
        'long': long
    }
    return pd.DataFrame(data, index=[0])

  
