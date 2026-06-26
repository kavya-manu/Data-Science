import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import json

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header{
    font-size:3rem;
    color:#6a0dad;
    text-align:center;
    margin-bottom:2rem;
}
.prediction-card{
    background-color:#f0f8ff;
    padding:2rem;
    border-radius:10px;
    border-left:5px solid #6a0dad;
    margin:1rem 0;
}
.confidence-bar{
    height:20px;
    background:#e0e0e0;
    border-radius:10px;
    margin:5px 0;
}
.confidence-fill{
    height:100%;
    border-radius:10px;
    background:linear-gradient(90deg,#ff6b6b,#4ecdc4);
    color:white;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model(format_type="joblib"):
    try:
        if format_type == "joblib":
            return joblib.load("models/iris_model.joblib")
        else:
            with open("models/iris_model.pickle", "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_model_info():
    try:
        with open("models/model_info.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None


@st.cache_resource
def load_feature_ranges():
    try:
        with open("models/feature_ranges.json", "r") as f:
            return json.load(f)
    except:
        return {
            "sepal_length":{"min":4.0,"max":8.0,"default":5.8},
            "sepal_width":{"min":2.0,"max":4.5,"default":3.0},
            "petal_length":{"min":1.0,"max":7.0,"default":4.0},
            "petal_width":{"min":0.1,"max":2.5,"default":1.2}
        }


model_info = load_model_info()
feature_ranges = load_feature_ranges()
model = load_model("joblib")


# Sidebar
with st.sidebar:

    st.title("⚙️ Settings")

    model_format = st.radio(
        "Model Format",
        ["joblib","pickle"]
    )

    if st.button("🔄 Reload Model"):
        model = load_model(model_format)
        if model:
            st.success(f"Loaded {model_format} model successfully.")

    st.divider()

    st.subheader("📊 Model Information")

    if model_info:
        st.write(f"**Type:** {model_info['model_type']}")
        st.write(f"**Accuracy:** {model_info['accuracy']:.1%}")
        st.write(f"**Features:** {len(model_info['feature_names'])}")
        st.write(f"**Classes:** {len(model_info['target_names'])}")

    st.divider()

    st.subheader("🚀 Quick Actions")

    if st.button("📊 Show Dataset Info"):
        st.session_state.show_dataset_info=True

    if st.button("🎯 Make Prediction"):
        st.session_state.make_prediction=True
# ==========================
# Main Content
# ==========================

st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

st.markdown("""
This app predicts the species of an Iris flower based on its measurements using a machine learning model.

Adjust the sliders below to input the flower's characteristics and click **Predict Species**.
""")

col1, col2 = st.columns([2, 1])

# ==========================
# Left Column
# ==========================
with col1:

    st.header("📝 Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges["sepal_length"]["min"]),
        max_value=float(feature_ranges["sepal_length"]["max"]),
        value=(float(feature_ranges["sepal_length"]["min"]) + float(feature_ranges["sepal_length"]["max"])) / 2,
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges["sepal_width"]["min"]),
        max_value=float(feature_ranges["sepal_width"]["max"]),
        value=(float(feature_ranges["sepal_width"]["min"]) + float(feature_ranges["sepal_width"]["max"])) / 2,
        step=0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges["petal_length"]["min"]),
        max_value=float(feature_ranges["petal_length"]["max"]),
        value=(float(feature_ranges["petal_length"]["min"]) + float(feature_ranges["petal_length"]["max"])) / 2,
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges["petal_width"]["min"]),
        max_value=float(feature_ranges["petal_width"]["max"]),
        value=(float(feature_ranges["petal_width"]["min"]) + float(feature_ranges["petal_width"]["max"])) / 2,
        step=0.1
    )

# ==========================
# Right Column
# ==========================
with col2:

    st.header("📊 Current Values")

    features_df = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value (cm)": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(
        features_df,
        hide_index=True,
        use_container_width=True
    )

# ==========================
# Prediction Input
# ==========================

input_features = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])
# =====================================
# Prediction
# =====================================

if st.button("🎯 Predict Species", type="primary", use_container_width=True):

    if model is not None and model_info is not None:

        try:
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)[0]

            predicted_class = model_info["target_names"][prediction[0]]

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.subheader("📋 Prediction Result")

            st.success(f"🌸 Predicted Species: **{predicted_class}**")

            st.subheader("📈 Confidence Scores")

            for i, probability in enumerate(prediction_proba):

                species = model_info["target_names"][i]
                percentage = probability * 100

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.progress(float(probability))

                with col2:
                    st.write(f"{percentage:.1f}%")

                st.write(f"**{species}**")

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

    else:
        st.error("❌ Model could not be loaded. Please check the model files.")


# =====================================
# About Dataset
# =====================================

with st.expander("📚 About the Iris Dataset"):

    st.markdown("""
### Iris Flower Dataset

The Iris flower dataset is one of the most famous datasets in machine learning, introduced by **Ronald A. Fisher** in 1936.

### Dataset Characteristics
- **150 samples** (50 per class)
- **4 features** per sample
- **3 classes (species)**

### Species
- 🌸 Iris Setosa
- 🌸 Iris Versicolor
- 🌸 Iris Virginica

### Features
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

This application uses a **Random Forest Classifier** with an accuracy of approximately **96%**.
""")


# =====================================
# Footer
# =====================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">
        <p>🌸 Built with <b>Streamlit</b> and <b>Scikit-learn</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
