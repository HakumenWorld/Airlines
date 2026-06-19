import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", page_icon="✈️", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "flight_price_model.pkl")

@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "model_metadata.json", "r") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

st.title("✈️ Prediksi Harga Tiket Pesawat")
st.write("Aplikasi ini memprediksi harga tiket pesawat berdasarkan maskapai, rute, waktu, durasi, kelas, dan sisa hari sebelum keberangkatan.")

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
    source_city = st.selectbox("Source City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
    departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])

with col2:
    arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    destination_city = st.selectbox("Destination City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
    flight_class = st.selectbox("Class", ["Economy", "Business"])
    duration = st.number_input("Duration (hours)", min_value=0.5, max_value=60.0, value=2.5, step=0.25)
    days_left = st.number_input("Days Left", min_value=1, max_value=60, value=15, step=1)

input_data = pd.DataFrame([{
    "airline": airline,
    "source_city": source_city,
    "departure_time": departure_time,
    "stops": stops,
    "arrival_time": arrival_time,
    "destination_city": destination_city,
    "class": flight_class,
    "duration": duration,
    "days_left": days_left,
}])

if st.button("Prediksi Harga"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimasi harga tiket: ₹{prediction:,.0f}")
    st.caption("Prediksi berasal dari model Decision Tree Regressor yang dilatih pada dataset Airlines Flights Data.")

with st.expander("Lihat data input"):
    st.dataframe(input_data)
