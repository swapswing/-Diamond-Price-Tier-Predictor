import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# Streamlit page config
# =============================
st.set_page_config(
    page_title="Diamond Price Tier Predictor",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Tier Predictor (Low / Medium / High)")
st.write(
    "This app loads a pre-trained model from `model.pkl` and uses the same "
    "pipeline you used in the notebook: dict → DataFrame → one-hot encoding "
    "→ reindex → scaling → prediction."
)

# =============================
# 1. Load artifacts from model.pkl (fixed)
# =============================
@st.cache_resource
def load_artifacts(pkl_path: str = "model.pkl"):
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)

    # Ensure we actually got a dict
    if not isinstance(artifacts, dict):
        raise TypeError(
            "model.pkl does not contain the expected artifacts dictionary.\n"
            "Re-save it as a dict with keys: "
            "['model', 'scaler', 'label_encoder', 'feature_columns', 'numeric_cols']."
        )

    required_keys = [
        "model",
        "scaler",
        "label_encoder",
        "feature_columns",
        "numeric_cols",
    ]
    missing = [k for k in required_keys if k not in artifacts]
    if missing:
        raise ValueError(
            f"`model.pkl` is missing keys: {missing}. "
            "Re-save it with all required artifacts after training."
        )

    return artifacts

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_columns = artifacts["feature_columns"]
numeric_cols = artifacts["numeric_cols"]

# =============================
# 2. Prediction helper
# =============================
def predict_price_category(
    trained_model,
    scaler,
    label_encoder,
    feature_cols,
    numeric_cols,
    new_data: dict,
):
    """
    Predict price category (Low / Medium / High) for a single diamond.

    Steps:
      1. dict -> pandas DataFrame
      2. one-hot encode cut, color, clarity
      3. reindex to match training feature_cols
      4. scale numeric_cols with the fitted scaler
      5. predict with trained_model
      6. decode class with label_encoder
    """

    # 1. Dict -> DataFrame
    input_df = pd.DataFrame([new_data])

    # 2. One-hot encode categorical features to match training
    input_df = pd.get_dummies(
        input_df,
        columns=["cut", "color", "clarity"],
        drop_first=True,
    )

    # 3. Align columns with training features
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # 4. Scale numeric features
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # 5. Predict and decode label
    pred_encoded = trained_model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label

# =============================
# 3. Streamlit UI for user input
# =============================
st.subheader("Enter diamond features")

carat = st.number_input("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
depth = st.number_input("Depth (%)", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
table = st.number_input("Table (%)", min_value=43.0, max_value=95.0, value=57.0, step=0.5)
x = st.number_input("Length x (mm)", min_value=3.0, max_value=11.0, value=6.5, step=0.01)
y = st.number_input("Width y (mm)", min_value=3.0, max_value=11.0, value=6.5, step=0.01)
z = st.number_input("Depth z (mm)", min_value=2.0, max_value=8.0, value=4.0, step=0.01)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
)

user_input = {
    "carat": float(carat),
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": float(depth),
    "table": float(table),
    "x": float(x),
    "y": float(y),
    "z": float(z),
}

if st.button("Predict price tier"):
    pred = predict_price_category(
        trained_model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_cols=feature_columns,
        numeric_cols=numeric_cols,
        new_data=user_input,
    )
    st.success(f"Predicted Price Tier: **{pred}**")
    with st.expander("See input as JSON"):
        st.json(user_input)
