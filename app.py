import streamlit as st
import pandas as pd
import joblib
import datetime

# Load Model & Column Order
pipeline = joblib.load("car_price_predictor.pkl")
trained_columns = joblib.load("trained_columns.pkl") 
st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to predict its selling price.")

# Input Widgets
year = st.number_input("Year", min_value=2000, max_value=2020, value=2014)
current_year = datetime.datetime.now().year
car_age = current_year - year
st.write(f"🕒 Car Age: **{car_age} years**")

km_driven = st.number_input("Kilometers Driven", min_value=1000, max_value=300000, value=70000, step=1000)
mileage = st.number_input("Mileage (km/l)", min_value=8.0, max_value=35.0, value=19.4, step=0.1)
engine = st.number_input("Engine (cc)", min_value=600, max_value=4000, value=1248, step=50)
max_power = st.number_input("Max Power (bhp)", min_value=40.0, max_value=400.0, value=82.0, step=1.0)
seats = st.number_input("Seats", min_value=2, max_value=9, value=5)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
brand = st.text_input("Brand", "Maruti")
price_per_km = st.number_input("Price per km", min_value=0.1, max_value=10.0, value=2.5)


# Prepare Data
input_df = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats,
    "car_age": current_year - year,
    "price_per_km": price_per_km,
    "brand": brand
}])

# One-Hot Encode & Reindex
input_df = pd.get_dummies(input_df, columns=['fuel','seller_type','transmission','owner','brand'], drop_first=True)
input_df = input_df.reindex(columns=trained_columns, fill_value=0)

# Predict
if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"💰 Estimated Price: {prediction:,.2f}")
