import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

rfc = joblib.load('rfc_model.joblib')
scaler = joblib.load('scaler.joblib')

def Inputs():
  st.sidebar.header = ("Inputs For Prediction")
  
