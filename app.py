import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

rfc = joblib.load('rfc_model.joblib')
scaler = joblib.load('scaler.joblib')

def Inputs():
  st.sidebar.header("Inputs For Prediction")

  temp = st.sidebar.slider('Temperature (C*)', 15, 35, 16)
  humidity = st.sidebar.slider('Humidity (%)', 0, 100, 1)
  wind_speed = st.sidebar.slider('wind_Speed (Km/h)', 0, 50, 1)
  LDSR = st.sidebar.slider('Last Day Since Rain)', 0, 7, 1)

  st.sidebar.subheader("Location Inputs")
  lat = st.sidebar.slider('Latitude', 50, 59, 55)
  long = st.sidebar.slider('Longitude', -124, -113, -118)

  data = {
        'temp': temp, 
        'humidity': humidity, 
        'wind_speed': wind_speed,
        'LDSR': LDSR,
        'lat': lat, 
        'long': long
        }
  return pd.DataFrame(data, index=[0])

st.title("Flame Cast")
input_df = Inputs()

if st.button('Predict Wildfire Within Area'):
  prediction_data = input_df[['temp', 'humidity', 'wind_speed', 'LDSR']]
  scaled_input = scaler.transform(prediction_data)

  prediction = rfc.predict(scaled_input)

  prediction_label = {0: 'Low Chance of Fire', 1: 'High Chance of Fire'}

  st.subheader("Predicted Result:")
  fire_risk_label = "High" if prediction == 1 else "Low"

  if fire_risk_label == "High":
      st.error(f"Predicted Fire Risk: **{fire_risk_label}**")
  else:
     st.success(f"Predicted Fire Risk: **{fire_risk_label}**")
            
  

  
