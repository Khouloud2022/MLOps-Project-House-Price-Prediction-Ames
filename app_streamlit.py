import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Titre principal
st.set_page_config(page_title="Prédiction Prix Maison - Ames", layout="centered")
st.title("Prédiction du Prix de Maison à Ames, Iowa")
st.markdown("---")

# Chargement du modèle et préprocesseur
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.success("Modèle RandomForest chargé - RMSE: 28 208 $")

st.markdown("### Entrez les caractéristiques de la maison")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Surface habitable (Gr Liv Area)", min_value=500, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Surface sous-sol (Total Bsmt SF)", min_value=0, max_value=6000, value=1000)
    year_built = st.number_input("Année de construction", min_value=1870, max_value=2025, value=2000)
    lot_area = st.number_input("Superficie terrain (Lot Area)", min_value=1300, max_value=215000, value=8450)
    overall_qual = st.slider("Qualité globale (Overall Qual)", 1, 10, 7)
    full_bath = st.selectbox("Salles de bain complètes", [1, 2, 3, 4], index=1)

with col2:
    neighborhood = st.selectbox("Quartier", [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", 
        "Gilbert", "NridgHt", "Sawyer", "NWAmes", "BrkSide"
    ])
    ms_zoning = st.selectbox("Zonage", ["RL", "RM", "FV", "RH", "C (all)"])
    kitchen_qual = st.selectbox("Qualité cuisine", ["TA", "Gd", "Ex", "Fa"], index=1)
    exter_qual = st.selectbox("Qualité extérieure", ["TA", "Gd", "Ex", "Fa"], index=0)
    heating_qc = st.selectbox("Qualité chauffage", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    central_air = st.selectbox("Climatisation centrale", ["Y", "N"], index=0)

if st.button("Prédire le prix de la maison", type="primary"):
    # Création du DataFrame
    input_data = pd.DataFrame([{
        'Gr Liv Area': gr_liv_area,
        'Total Bsmt SF': total_bsmt_sf,
        'Year Built': year_built,
        'Lot Area': lot_area,
        'Overall Qual': overall_qual,
        'Overall Cond': 5,
        'Full Bath': full_bath,
        'TotRms AbvGrd': 7,
        'Garage Cars': 2,
        'Garage Area': 500,
        '1st Flr SF': 1000,
        '2nd Flr SF': 500,
        'Neighborhood': neighborhood,
        'MS Zoning': ms_zoning,
        'Sale Condition': 'Normal',
        'Kitchen Qual': kitchen_qual,
        'Exter Qual': exter_qual,
        'Heating QC': heating_qc,
        'Central Air': central_air,
        'Foundation': 'PConc'
    }])

    # Prédiction
    X_processed = preprocessor.transform(input_data)
    prediction = model.predict(X_processed)[0]

    st.markdown("---")
    st.markdown(f"### Prix estimé de la maison")
    st.markdown(f"<h1 style='text-align: center; color: #1E90FF;'>${prediction:,.0f}</h1>", unsafe_allow_html=True)
    
    if prediction > 250000:
        st.balloons()
        st.success("Maison de luxe détectée !")
    elif prediction > 150000:
        st.success("Belle maison dans la moyenne haute")
    else:
        st.info("Maison abordable")

st.markdown("---")
st.caption("Projet MLOps complet - Git + DVC + MLflow + FastAPI + Docker + Streamlit - Khouloud Ouni et Eya BenKhadhra")