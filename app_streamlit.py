import streamlit as st
import requests
import json

# ─────────────────────────────────────────────────────────────
# Page config + Design luxueux
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="house",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS pour rendre ça magnifique
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #1E90FF; color: white; font-weight: bold; border-radius: 10px; height: 3em; width: 100%;}
    .prediction {font-size: 48px; font-weight: bold; text-align: center; color: #1E90FF;}
    .header {font-size: 42px; text-align: center; color: #1E90FF; font-weight: bold;}
    .footer {text-align: center; margin-top: 50px; color: #666;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Titre principal
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 class='header'>Ames House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>MLOps Project • Khouloud Ouni & Eya Ben Khadhra</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Sidebar avec info projet
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d2/Font_Awesome_5_solid_home_white.svg", width=100)
    st.markdown("### Model Performance")
    st.success("Random Forest\nRMSE:  28207.582545732184")
    st.info("Trained on Ames Housing Dataset\n2,930 properties • 79 features")
    st.markdown("### Tech Stack")
    st.write("• FastAPI • DVC • MLflow\n• Docker • GitHub Actions\n• Streamlit • Evidently AI")

# ─────────────────────────────────────────────────────────────
# Inputs en deux colonnes
# ─────────────────────────────────────────────────────────────
st.markdown("### Enter House Features")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 6000, 1710)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 6000, 856)
    year_built = st.number_input("Year Built", 1870, 2025, 2003)
    lot_area = st.number_input("Lot Area (sq ft)", 1300, 215000, 8450)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)

with col2:
    neighborhood = st.selectbox("Neighborhood", [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
        "Gilbert", "NridgHt", "Sawyer", "NWAmes", "BrkSide", "Crawfor", "NoRidge"
    ])
    kitchen_qual = st.selectbox("Kitchen Quality", ["TA", "Gd", "Ex", "Fa"], index=1)
    exter_qual = st.selectbox("Exterior Quality", ["TA", "Gd", "Ex", "Fa"], index=0)
    heating_qc = st.selectbox("Heating Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=0)
    central_air = st.selectbox("Central Air", ["Y", "N"], index=0)

# ─────────────────────────────────────────────────────────────
# Bouton de prédiction + appel API
# ─────────────────────────────────────────────────────────────
if st.button("Predict House Price", type="primary", use_container_width=True):
    with st.spinner("Contacting prediction API..."):
        payload = {
            "Gr_Liv_Area": float(gr_liv_area),
            "Total_Bsmt_SF": float(total_bsmt_sf),
            "Year_Built": int(year_built),
            "Lot_Area": int(lot_area),
            "Overall_Qual": int(overall_qual),
            "Overall_Cond": 5,
            "Full_Bath": int(full_bath),
            "TotRms_AbvGrd": 8,
            "Garage_Cars": 2,
            "Garage_Area": 548,
            "First_Flr_SF": 856,
            "Second_Flr_SF": 854,
            "Neighborhood": neighborhood,
            "MS_Zoning": "RL",
            "Sale_Condition": "Normal",
            "Kitchen_Qual": kitchen_qual,
            "Exter_Qual": exter_qual,
            "Heating_QC": heating_qc,
            "Central_Air": central_air,
            "Foundation": "PConc"
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                price = result["predicted_sale_price"]
                
                st.markdown("---")
                st.markdown(f"<h2 class='prediction'>${price:,.0f}</h2>", unsafe_allow_html=True)
                
                if price > 300000:
                    st.balloons()
                    st.success("Luxury Property Detected!")
                elif price > 200000:
                    st.success("Premium Property")
                elif price > 150000:
                    st.info("Great Family Home")
                else:
                    st.info("Affordable Housing")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except:
            st.error("Cannot reach API. Make sure FastAPI is running on port 8000")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class='footer'>
    <b>MLOps Project • Khouloud Ouni & Eya Ben Khadhra • December 2025</b><br>
    
</div>
""", unsafe_allow_html=True)