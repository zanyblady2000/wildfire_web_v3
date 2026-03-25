import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import numpy as np

np.random.seed(42)

rfc = joblib.load('rfc_model.joblib')
scaler = joblib.load('scaler.joblib')

def Inputs():
  st.sidebar.header("Inputs For Prediction")

  temp = st.sidebar.slider('Temperature (C*)', 15, 35, 20)
  humidity = st.sidebar.slider('Humidity (%)', 0, 100, 35)
  windspeed = st.sidebar.slider('windspeed (Km/h)', 0, 50, 15)
  LDSR = st.sidebar.slider('Last Day Since Rain)', 0, 7, 3)

  st.sidebar.subheader("Location Inputs")
  lat = st.sidebar.slider('Latitude', 50, 59, 55)
  long = st.sidebar.slider('Longitude', -124, -113, -118)

  data = {
        'temp': temp, 
        'humidity': humidity, 
        'windspeed': windspeed,
        'LDSR': LDSR,
        'lat': lat, 
        'long': long
        }

st.title("Flame Cast")
input_df = Inputs()

tab1, tab2 = st.tabs(["August", "September"])

with tab1:
  st.header("Data For August")

  Aug_data = {
    "temp": np.random.randint(15, 35, size=31),
    "humidity": np.random.randint(0, 100, size=31),
    "windspeed": np.random.randint(0, 50, size=31),
    "LDSR": np.random.randint(0, 7, size=31)
  }

  Aug_df = pd.DataFrame(Aug_data)

  Aug_df.index = np.arange(1, 32)
  Aug_df.index.name = "Day"

  Aug_prediction_data = Aug_df[['temp', 'humidity', 'windspeed', 'LDSR']]
  Aug_scaled_input = scaler.transform(Aug_prediction_data)
  Aug_prediction = rfc.predict(Aug_scaled_input)

  Aug_prediction_label = {0: 'High Risk', 1: 'Low Risk'}

  Aug_df["Aug_prediction"] = Aug_prediction 
  
  order = ["temp", "humidity", "windspeed", "LDSR", "Aug_prediction"]
  Aug_df = Aug_df[order]

  st.table(Aug_df)

with tab2:
  st.header("Data For September")

  Sept_data = {
    "temp": np.random.randint(15, 35, size=31),
    "humidity": np.random.randint(0, 100, size=31),
    "windspeed": np.random.randint(0, 50, size=31),
    "LDSR": np.random.randint(0, 7, size=31)
  }

  Sept_df = pd.DataFrame(Sept_data)

  Sept_df.index = np.arange(1, 32)
  Sept_df.index.name = "Day"

  Sept_prediction_data = Sept_df[['temp', 'humidity', 'windspeed', 'LDSR']]
  Sept_scaled_input = scaler.transform(Sept_prediction_data)
  Sept_prediction = rfc.predict(Sept_scaled_input)

  Sept_prediction_label = {0: 'High Risk', 1: 'Low Risk'}

  Sept_df["Sept_prediction"] = Sept_prediction 
  
  order_2 = ["temp", "humidity", "windspeed", "LDSR", "Sept_prediction"]
  Sept_df = Sept_df[order_2]

  st.table(Sept_df)
  
if st.button('Predict Wildfire Within Area'):
  prediction_data = input_df[['temp', 'humidity', 'windspeed', 'LDSR']]
  scaled_input = scaler.transform(prediction_data)

  prediction = rfc.predict(scaled_input)

  prediction_label = {0: 'Low Chance of Fire', 1: 'High Chance of Fire'}

  st.subheader("Predicted Result:")
  fire_risk_label = "High" if prediction[0] == 1 else "Low"

  if fire_risk_label == "High":
      st.error(f"Predicted Fire Risk: **{fire_risk_label}**")
  else:
     st.success(f"Predicted Fire Risk: **{fire_risk_label}**")

  map_df = input_df.copy()
  map_df['prediction_label'] = fire_risk_label[prediction.item()]
        
       
  fig = px.scatter_mapbox(
            map_df, 
            lat="lat",
            lon="long", 
            color="prediction_label", 
            color_discrete_map={fire_risk_label[1]: 'red', fire_risk_label[0]: 'green'},
            zoom=8, 
            center={"lat": map_df['lat'].iloc[0], "lon": map_df['long'].iloc[0]},
            height=500,
            mapbox_style="carto-positron", 
            hover_data=['temp', 'humidity', 'windspeed']
        )
        
  st.subheader("Location Visualization")
  st.plotly_chart(fig, use_container_width=True)
            
  

  
