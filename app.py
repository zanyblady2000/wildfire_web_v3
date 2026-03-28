import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import numpy as np

np.random.seed(42) # Makes it so randomly generated numbers stay the same throughout every run.

rfc = joblib.load('rfc_model.joblib') # Using Joblib to import the trained ML model and scaler.
scaler = joblib.load('scaler.joblib')

def Inputs(): # Defining Sidebar prediction inputs and returning said inputs as a DataFrame.
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
  return pd.DataFrame(data, index=[0])

st.title(":red[Flame]:orange[Cast]", text_alignment="center")
st.subheader("The Wildfire Predictor", text_alignment="center")
input_df = Inputs()

st.subheader("About FlameCast")
st.write("Flamecast is an innovative application idea created to predict wildfires, which are a known problem throughout Canada and beyond and can be extremely unpredictable. Flamecast will predict the possibility of wildfires by using realtime weather data including temperature, wind speed, and last day since rain.")

st.header("Calendar")
tab1, tab2 = st.tabs(["August", "September"])

with tab1: 
  st.header("Data For August")

  Aug_rng = np.random.default_rng() # Defines a random number generator

  Aug_data = {
    "temp": np.random.randint(15, 35, size=31), # randint = random integer and size=31 means it will generate only 31 different random integers
    "humidity": np.random.randint(0, 100, size=31),
    "windspeed": np.random.randint(0, 50, size=31),
    "LDSR": np.random.randint(0, 7, size=31),
    "lat": Aug_rng.uniform(50, 59, size=31), # Uses the Random number generator to generate decimal numbers
    "long": Aug_rng.uniform(-124, -113, size=31)
  }

  Aug_df = pd.DataFrame(Aug_data)

  Aug_df.index = np.arange(1, 32)
  Aug_df.index.name = "Day"

  Aug_prediction_data = Aug_df[['temp', 'humidity', 'windspeed', 'LDSR']] # Runs a prediction using the August dataset
  Aug_scaled_input = scaler.transform(Aug_prediction_data)
  Aug_prediction = rfc.predict(Aug_scaled_input)

  Aug_prediction_label = {0: 'Low Risk', 1: 'High Risk'}

  Aug_df["Aug_prediction"] = ["Low Risk" if p == 0 else "High Risk" for p in Aug_prediction]
  
  order = ["temp", "humidity", "windspeed", "LDSR", "lat", "long", "Aug_prediction"]
  Aug_df = Aug_df[order]

  st.table(Aug_df)

  Aug_map_df = Aug_df.copy()

  Aug_data_fig = px.scatter_mapbox( # Makes a visual map out of the Aug data to show every "Day" or row
    Aug_map_df.reset_index(),
    lat="lat",
    lon="long",
    zoom=4,
    color="Aug_prediction",
    color_discrete_map={"High Risk": 'red', "Low Risk": 'green'},
    height=500,
    hover_data=["Day"],
    mapbox_style="open-street-map"
  )

  
  st.title("August Fire Risk Map")
  st.plotly_chart(Aug_data_fig)
  
with tab2: # Everything in the September tab is the same as in the August tab.
  st.header("Data For September")

  Sept_rng = np.random.default_rng()

  Sept_data = {
    "temp": np.random.randint(15, 35, size=31),
    "humidity": np.random.randint(0, 100, size=31),
    "windspeed": np.random.randint(0, 50, size=31),
    "LDSR": np.random.randint(0, 7, size=31),
    "lat": Sept_rng.uniform(50, 59, size=31),
    "long": Sept_rng.uniform(-124, -113, size=31)
  }

  Sept_df = pd.DataFrame(Sept_data)

  Sept_df.index = np.arange(1, 32)
  Sept_df.index.name = "Day"

  Sept_prediction_data = Sept_df[["temp", "humidity", "windspeed", "LDSR"]]
  Sept_scaled_input = scaler.transform(Sept_prediction_data)
  Sept_prediction = rfc.predict(Sept_scaled_input)

  Sept_prediction_label = {0: 'Low Risk', 1: 'High Risk'}

  Sept_df["Sept_prediction"] = ["Low Risk" if p == 0 else "High Risk" for p in Sept_prediction]
  
  order_2 = ["temp", "humidity", "windspeed", "LDSR", "lat", "long", "Sept_prediction"]
  Sept_df = Sept_df[order_2]

  st.table(Sept_df)

  Sept_map_df = Sept_df.copy()

  Sept_data_fig = px.scatter_mapbox(
    Sept_map_df.reset_index(),
    lat="lat",
    lon="long",
    zoom=4,
    color="Sept_prediction",
    color_discrete_map={"High Risk": 'red', "Low Risk": 'green'},
    height=500,
    hover_data=["Day"],
    mapbox_style="open-street-map"
  )

  
  st.title("September Fire Risk Map")
  st.plotly_chart(Sept_data_fig)

st.header("Sidebar Prediction Scenario")
  
if st.button('Predict Wildfire In Nearby Area'): # Uses the showable sidebar inputs to run a prediction.
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
  map_df['fire_risk_label'] = fire_risk_label
        
       
  fig = px.scatter_mapbox( # Maps out the Sidebar scenario prediction
            map_df, 
            lat="lat",
            lon="long", 
            color="fire_risk_label", 
            color_discrete_map={"High": 'red', "Low": 'green'}
            zoom=8, 
            height=500,
            mapbox_style="open-street-map", 
            hover_data=['temp', 'humidity', 'windspeed']
        )
        
  st.subheader("Location Visualization")
  st.plotly_chart(fig, use_container_width=True)
            
  

  
