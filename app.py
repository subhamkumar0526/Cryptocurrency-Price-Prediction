import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide"
)

# Custom design (only colors changed)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #3a1c71, #d76d77, #ffaf7b);
    background-attachment: fixed;
    color: white;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: #ffffff !important;
}

/* Colorful title */
h1 {
    background: linear-gradient(90deg, #ffd200, #ff6a88, #00f5d4, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2980, #7b2ff7, #f107a3);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #fff200 !important;
    -webkit-text-fill-color: #fff200;
    background: none;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #ffffff !important;
}

/* Cards */
.card {
    background: linear-gradient(145deg, #11998e, #38ef7d);
    border: 2px solid #ffffff55;
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 15px;
}

.card h3 {
    color: #0b3d3a !important;
    font-size: 16px;
}

.card h2 {
    color: #ffffff !important;
    font-size: 25px;
}

.card p {
    color: #0b3d3a !important;
}

/* Second card in a different color */
div[data-testid="column"]:nth-of-type(2) .card {
    background: linear-gradient(145deg, #fc466b, #3f5efb);
}

div[data-testid="column"]:nth-of-type(2) .card h3,
div[data-testid="column"]:nth-of-type(2) .card p {
    color: #ffe9f0 !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #f09819, #dd2476);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #00c9a7, #00b4db, #7b2ff7);
    color: white;
    border: 1px solid white;
}

/* Result */
.result {
    background: linear-gradient(100deg, #f7971e, #ffd200, #00e676);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.result p {
    color: #2b2b2b !important;
    font-weight: bold;
}

.result h2 {
    color: #1b1b1b !important;
    font-size: 32px;
}

.footer {
    text-align: center;
    color: #fff3c4;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# Load trained model
try:
    with open("random_forest_model.pkl", "rb") as file:
        model_rf = pickle.load(file)

except FileNotFoundError:
    st.error("Model file not found: random_forest_model.pkl")
    st.stop()


# Prediction function
def predict_btc_price(input_data):
    prediction = model_rf.predict(input_data)
    return prediction[0]


# Main application
def main():

    # Simple heading: no banner, no AI title
    st.title("₿ Bitcoin Price Predictor")
    st.caption("Machine Learning Dashboard | Random Forest Regressor")

    # Only two dashboard cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>TARGET ASSET</h3>
            <h2>Bitcoin</h2>
            <p>BTC-USD</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>MODEL</h3>
            <h2>Random Forest</h2>
            <p>Price prediction</p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar inputs
    st.sidebar.title("Input Features")
    st.sidebar.write("Enter cryptocurrency data below.")

    usdt_close = st.sidebar.number_input(
        "USDT Close Price",
        min_value=0.0,
        value=1.0,
        format="%.2f"
    )

    usdt_volume = st.sidebar.number_input(
        "USDT Volume",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    bnb_close = st.sidebar.number_input(
        "BNB Close Price",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    bnb_volume = st.sidebar.number_input(
        "BNB Volume",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

    # Create input dataframe
    input_data = pd.DataFrame({
        "USDT_Close": [usdt_close],
        "USDT_Volume": [usdt_volume],
        "BNB_Close": [bnb_close],
        "BNB_Volume": [bnb_volume]
    })

    st.subheader("Input Data")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

    # Prediction button: no balloons or animations
    if st.button("Predict BTC Close Price"):
        try:
            predicted_price = predict_btc_price(input_data)

            st.markdown(f"""
            <div class="result">
                <p>PREDICTED BITCOIN CLOSING PRICE</p>
                <h2>${predicted_price:,.2f} USD</h2>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("""
    <div class="footer">
        Built with Python | Streamlit | Pandas | Scikit-learn
        <br>
        Cryptocurrency predictions are estimates, not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
