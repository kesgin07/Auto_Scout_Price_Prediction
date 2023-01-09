
import pandas as pd
import numpy as np
import pickle
import streamlit as st



st.header("Welcome to Auto Price Prediction")

df = pd.read_csv('auto_scout_for_deployment.csv')
st.write(df.head())




st.sidebar.title("_Please Enter the Features to Predict the Price of Auto_")

make_model = st.sidebar.selectbox('Make & Model', (df.make_model.unique()))
fuels = st.sidebar.selectbox('Fuel', (df.Fuel.unique()))
hp = st.sidebar.selectbox('HP', (sorted(df.hp_kW.unique())))   
gearing_type = st.sidebar.selectbox('Gearing Type', sorted(df.Gearing_Type.unique())) 
gear = st.sidebar.number_input("Gears", min_value=5, max_value=8, value=5, step=1)
age = st.sidebar.number_input("Age", min_value=0, max_value=3, value=0, step=1)
km = st.sidebar.slider("KM",  min_value=0, max_value=320000, value=50000, step=1000)




st.write("Your Car Features")
my_dict = {
    "make_model": make_model,
    "hp_kW": hp,
    "km": km,
    'age': age,
    'Gearing_Type': gearing_type,
    'Gears':gear,
    'Fuel':fuels
}
df_input = pd.DataFrame.from_dict([my_dict])

st.write(df_input.head())



model = pickle.load(open("model_auto", "rb"))


prediction = round(model.predict(df_input)[0], 2)
prediction = str(prediction) + " $" 
if st.button("Predict"):
    st.success(prediction) 
