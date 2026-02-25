import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Predicción IMDB Rating", page_icon="🎬")

st.title("🎬 Predicción de IMDB Rating")
st.write("Introduce datos y el modelo estima el **IMDB_Rating**.")

MODEL_PATH = os.path.join("models", "modelo.pkl")

@st.cache_resource
def cargar_modelo():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

modelo = cargar_modelo()

st.subheader("Entradas")
duracion = st.number_input("Runtime (min)", min_value=1.0, value=120.0, step=1.0)
votos = st.number_input("No_of_Votes", min_value=0.0, value=500000.0, step=1000.0)
meta = st.number_input("Meta_score", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

if st.button("Predecir"):
    X = np.array([[duracion, votos, meta]])
    pred = float(modelo.predict(X)[0])
    st.success(f"⭐ Predicción IMDB_Rating: {pred:.2f}")