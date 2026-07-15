import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Custom CSS for UI beautification
# Updated Custom CSS for UI beautification
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; padding: 20px; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSlider > div[data-baseweb="slider"] {
        padding: 6px;
    }
    .stDataFrame { 
        border-radius: 8px; 
        overflow: hidden; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    h1, h2, h3 { color: #343a40; font-family: 'Arial', sans-serif; }
    .stMarkdown { font-family: 'Arial', sans-serif; color: #495057; }
    .plot-container { 
        border: 1px solid #dee2e6; 
        border-radius: 8px; 
        padding: 10px; 
        background-color: #f1f3f5; 
    }
    .success-box { 
        background-color: #28a745; 
        color: #ffffff; 
        padding: 10px; 
        border-radius: 8px; 
        margin: 10px 0; 
    }
    .error-box { 
        background-color: #dc3545; 
        color: #ffffff; 
        padding: 10px; 
        border-radius: 8px; 
        margin: 10px 0; 
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Pavement Distress Predictor")
st.sidebar.markdown("""
    This app predicts pavement distress metrics (PCR and Rut Average) using machine learning models.  
    Navigate through the tabs to:
    - Predict PCR or Rut Average using sliders.
    - Upload Excel files for batch predictions.
    - Visualize actual vs. predicted values.
    - Preprocess data by filling NaNs.
""")

@st.cache_resource
def load_model(model_name, target):
    try:
        with open(f"{target}_{model_name}_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{target}_{model_name}_model.pkl' not found.", icon="🚨")
        return None

model_names = ["random_forest", "gradient_boosting", "svm", "xgboost"]
pcr_models = {name: load_model(name, "INDEX_PCR") for name in model_names}
rut_models = {name: load_model(name, "RUT_AVG") for name in model_names}

st.title("🛣️ Pavement Distress Predictor")
st.markdown("Predict pavement distress metrics (PCR and Rut Average) using initial data. Select a tab below to get started.", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Predict PCR", "📏 Predict Rut", "📂 Excel Predictor", "📈 Visualize Data", "🧹 Preprocess Data"])

input_features = {
    "INDEX_PCR": [
        {"name": "INDEX_IRI", "label": "IRI Index", "min": 0.0, "max": 5.0, "default": 4.0, "step": 0.1, "help": "Index of International Roughness Index"},
        {"name": "IRI_AVG", "label": "Average IRI (m/km)", "min": 0.0, "max": 200.0, "default": 70.0, "step": 1.0, "help": "Average International Roughness Index in meters per kilometer"},
        {"name": "INDEX_RUT", "label": "Rut Index", "min": 0.0, "max": 5.0, "default": 3.5, "step": 0.1, "help": "Index of rutting severity"},
        {"name": "RUT_AVG", "label": "Average Rut Depth (mm)", "min": 0.0, "max": 0.5, "default": 0.15, "step": 0.01, "help": "Average depth of ruts in millimeters"},
        {"name": "PERCENT_CRACKING", "label": "Percent Cracking (%)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "Percentage of pavement surface with cracks"},
        {"name": "ALLIG_H", "label": "Alligator Cracking High (m²)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "High-severity alligator cracking area"},
        {"name": "ALLIG_M", "label": "Alligator Cracking Medium (m²)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "Medium-severity alligator cracking area"},
        {"name": "LONG_H", "label": "Longitudinal Cracking High (m)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "High-severity longitudinal cracking length"},
        {"name": "LONG_M", "label": "Longitudinal Cracking Medium (m)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "Medium-severity longitudinal cracking length"},
        {"name": "TRAN_H", "label": "Transverse Cracking High (m)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "High-severity transverse cracking length"},
        {"name": "TRAN_M", "label": "Transverse Cracking Medium (m)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "Medium-severity transverse cracking length"},
        {"name": "SPEED", "label": "Speed (km/h)", "min": 0.0, "max": 60.0, "default": 30.0, "step": 1.0, "help": "Average vehicle speed on pavement"},
        {"name": "WIDTH", "label": "Pavement Width (m)", "min": 10.0, "max": 15.0, "default": 11.5, "step": 0.1, "help": "Width of the pavement in meters"}
    ],
    "RUT_AVG": [
        {"name": "INDEX_IRI", "label": "IRI Index", "min": 0.0, "max": 5.0, "default": 4.0, "step": 0.1, "help": "Index of International Roughness Index"},
        {"name": "IRI_AVG", "label": "Average IRI (m/km)", "min": 0.0, "max": 200.0, "default": 70.0, "step": 1.0, "help": "Average International Roughness Index in meters per kilometer"},
        {"name": "INDEX_RUT", "label": "Rut Index", "min": 0.0, "max": 5.0, "default": 3.5, "step": 0.1, "help": "Index of rutting severity"},
        {"name": "RUT_AVG", "label": "Average Rut Depth (mm)", "min": 0.0, "max": 0.5, "default": 0.15, "step": 0.01, "help": "Average depth of ruts in millimeters"},
        {"name": "PERCENT_CRACKING", "label": "Percent Cracking (%)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "Percentage of pavement surface with cracks"},
        {"name": "ALLIG_H", "label": "Alligator Cracking High (m²)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "High-severity alligator cracking area"},
        {"name": "ALLIG_M", "label": "Alligator Cracking Medium (m²)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "Medium-severity alligator cracking area"},
        {"name": "LONG_H", "label": "Longitudinal Cracking High (m)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "High-severity longitudinal cracking length"},
        {"name": "LONG_M", "label": "Longitudinal Cracking Medium (m)", "min": 0.0, "max": 100.0, "default": 0.0, "step": 1.0, "help": "Medium-severity longitudinal cracking length"},
        {"name": "TRAN_H", "label": "Transverse Cracking High (m)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "High-severity transverse cracking length"},
        {"name": "TRAN_M", "label": "Transverse Cracking Medium (m)", "min": 0.0, "max": 50.0, "default": 0.0, "step": 0.1, "help": "Medium-severity transverse cracking length"},
        {"name": "SPEED", "label": "Speed (km/h)", "min": 0.0, "max": 60.0, "default": 30.0, "step": 1.0, "help": "Average vehicle speed on pavement"},
        {"name": "WIDTH", "label": "Pavement Width (m)", "min": 10.0, "max": 15.0, "default": 11.5, "step": 0.1, "help": "Width of the pavement in meters"}
    ]
}

def create_input_form(features, tab_key):
    input_values = {}
    st.markdown("**Input Parameters**")
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        with col:
            value = st.slider(
                feature["label"],
                min_value=float(feature["min"]),
                max_value=float(feature["max"]),
                value=float(feature["default"]),
                step=feature.get("step", 1.0),
                key=f"{feature['name']}_{tab_key}",
                help=feature.get("help", "")
            )
            input_values[feature["name"]] = value
    return pd.DataFrame([input_values], columns=[f["name"] for f in features])

def process_tab(tab, model_key, features, models=None):
    with tab:
        with st.container():
            st.subheader(f"🔍 Predict {model_key.replace('_', ' ').title()}")
            st.markdown(f"Adjust the sliders below to predict {model_key.replace('_', ' ').title()} using a selected model.")
            model_choice = st.selectbox("Choose a Model", options=model_names, key=f"model_{model_key}")
            input_data = create_input_form(features, model_key)
            st.markdown("**Preview of Input Parameters**")
            st.dataframe(input_data.style.format("{:.2f}"))
            if st.button(f"Predict {model_key.replace('_', ' ').title()}", key=f"predict_{model_key}"):
                with st.spinner("Generating prediction..."):
                    if models and models.get(model_choice):
                        prediction = models[model_choice].predict(input_data)[0]
                        st.markdown(f'<div class="success-box">Predicted {model_key.replace("_", " ").title()} ({model_choice.replace("_", " ").title()}): <b>{prediction:.2f}</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box">Model for {model_key.replace("_", " ").title()} not loaded. Check pickle files.</div>', unsafe_allow_html=True)

with tab3:
    with st.container():
        st.subheader("📂 Excel Predictor")
        st.markdown("Upload an Excel file to predict PCR or Rut Average and download the results.")
        col1, col2 = st.columns([1, 1])
        with col1:
            model_choice_excel = st.selectbox("Choose a Model", options=model_names, key="model_excel")
        with col2:
            target_choice = st.selectbox("Choose Prediction Target", options=["INDEX_PCR", "RUT_AVG"], key="target_excel")
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
        if uploaded_file:
            with st.spinner("Processing file..."):
                input_df = pd.read_excel(uploaded_file)
                required_cols = [f["name"] for f in input_features[target_choice]]
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                if missing_cols:
                    st.markdown(f'<div class="error-box">Missing columns in uploaded file: {", ".join(missing_cols)}. Expected columns: {", ".join(required_cols)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("**Uploaded Data Preview**")
                    st.dataframe(input_df[required_cols].style.format("{:.2f}"))
                    if st.button("Predict and Download", key="predict_excel"):
                        with st.spinner("Generating predictions..."):
                            models = pcr_models if target_choice == "INDEX_PCR" else rut_models
                            if models and models.get(model_choice_excel):
                                predictions = models[model_choice_excel].predict(input_df[required_cols])
                                output_df = input_df.copy()
                                output_df[f"{target_choice}"] = predictions
                                output_buffer = BytesIO()
                                output_df.to_excel(output_buffer, index=False)
                                output_buffer.seek(0)
                                st.markdown(f'<div class="success-box">Predictions generated successfully!</div>', unsafe_allow_html=True)
                                st.download_button(
                                    label="Download Predictions as Excel",
                                    data=output_buffer,
                                    file_name=f"{target_choice.lower()}_predictions.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.markdown(f'<div class="error-box">Selected model not loaded. Check pickle files.</div>', unsafe_allow_html=True)

with tab4:
    with st.container():
        st.subheader("📈 Data Visualization")
        st.markdown("Upload an Excel file to visualize actual vs. predicted values.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            vis_model = st.selectbox("Choose a Model", options=model_names, key="vis_model")
        with col2:
            vis_target = st.selectbox("Choose Prediction Target", options=["INDEX_PCR", "RUT_AVG"], key="vis_target")
        with col3:
            vis_type = st.selectbox("Choose Visualization Type", options=["Line Plot", "Scatter Plot"], key="vis_type")
        vis_file = st.file_uploader("Upload Excel File for Visualization", type=["xlsx"], key="vis_file")
        if vis_file:
            with st.spinner("Processing visualization..."):
                vis_df = pd.read_excel(vis_file)
                required_cols = [f["name"] for f in input_features[vis_target]]
                missing_cols = [col for col in required_cols if col not in vis_df.columns]
                if missing_cols:
                    st.markdown(f'<div class="error-box">Missing columns: {", ".join(missing_cols)}. Expected columns: {", ".join(required_cols)}</div>', unsafe_allow_html=True)
                else:
                    models = pcr_models if vis_target == "INDEX_PCR" else rut_models
                    if models and models.get(vis_model):
                        vis_df[f"{vis_target}"] = models[vis_model].predict(vis_df[required_cols])
                        actual_col = "INDEX_PCR" if vis_target == "INDEX_PCR" else "RUT_AVG"
                        st.markdown("**Visualization**")
                        with st.container():
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            if actual_col in vis_df.columns:
                                st.write(f"Actual vs Predicted {vis_target.replace('_', ' ').title()} (Averaged over 100 points)")
                                avg_old = vis_df[actual_col].groupby(vis_df.index // 100).mean()
                                avg_new = vis_df[f"{vis_target}"].groupby(vis_df.index // 100).mean()
                                fig, ax = plt.subplots(figsize=(8, 5))
                                if vis_type == "Line Plot":
                                    ax.plot(avg_old.index, avg_old, color="#007bff", label=f"Actual {vis_target.replace('_', ' ').title()}", linewidth=2)
                                    ax.plot(avg_new.index, avg_new, color="#ff6f00", label=f"Predicted {vis_target.replace('_', ' ').title()}", linewidth=2)
                                else:  # Scatter Plot
                                    ax.scatter(avg_old.index, avg_old, color="#007bff", label=f"Actual {vis_target.replace('_', ' ').title()}", alpha=0.6, s=50)
                                    ax.scatter(avg_new.index, avg_new, color="#ff6f00", label=f"Predicted {vis_target.replace('_', ' ').title()}", alpha=0.6, s=50)
                                ax.set_xlabel("Group Index (Averaged over 100 points)", fontsize=12)
                                ax.set_ylabel(f"{vis_target.replace('_', ' ').title()} Value", fontsize=12)
                                ax.legend(fontsize=10)
                                ax.grid(True, linestyle="--", alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.markdown(f'<div class="error-box">Column "{actual_col}" not found in uploaded file. Showing alternative visualization.</div>', unsafe_allow_html=True)
                                avg_feature = vis_df["IRI_AVG"].groupby(vis_df.index // 100).mean()
                                avg_new = vis_df[f"{vis_target}"].groupby(vis_df.index // 100).mean()
                                fig, ax = plt.subplots(figsize=(8, 5))
                                if vis_type == "Line Plot":
                                    ax.plot(avg_feature.index, avg_feature, color="#007bff", label="IRI Average", linewidth=2)
                                    ax.plot(avg_new.index, avg_new, color="#ff6f00", label=f"Predicted {vis_target.replace('_', ' ').title()}", linewidth=2)
                                else:  # Scatter Plot
                                    ax.scatter(avg_feature.index, avg_feature, color="#007bff", label="IRI Average", alpha=0.6, s=50)
                                    ax.scatter(avg_new.index, avg_new, color="#ff6f00", label=f"Predicted {vis_target.replace('_', ' ').title()}", alpha=0.6, s=50)
                                ax.set_xlabel("Group Index (Averaged over 100 points)", fontsize=12)
                                ax.set_ylabel("Value", fontsize=12)
                                ax.legend(fontsize=10)
                                ax.grid(True, linestyle="--", alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box">Selected model not loaded. Check pickle files.</div>', unsafe_allow_html=True)

with tab5:
    with st.container():
        st.subheader("🧹 Preprocess Data")
        st.markdown("Upload an Excel file to preprocess NaN values using interpolation (average of 5 rows above and below).")
        preprocess_file = st.file_uploader("Upload Excel File for Preprocessing", type=["xlsx"], key="preprocess_file")
        if preprocess_file:
            with st.spinner("Preprocessing data..."):
                preprocess_df = pd.read_excel(preprocess_file)
                st.markdown("**Original Data Preview (with NaNs)**")
                st.dataframe(preprocess_df.style.format("{:.2f}"))
                def interpolate_window(df, column, window=5):
                    interpolated = df[column].copy()
                    nan_indices = interpolated.isna()
                    for idx in nan_indices[nan_indices].index:
                        start = max(0, idx - window)
                        end = min(len(df), idx + window + 1)
                        window_values = df[column].iloc[start:end].dropna()
                        if not window_values.empty:
                            interpolated.iloc[idx] = window_values.mean()
                    return interpolated
                numeric_cols = preprocess_df.select_dtypes(include=[np.number]).columns
                preprocessed_df = preprocess_df.copy()
                for col in numeric_cols:
                    if preprocessed_df[col].isna().sum() > 0:
                        preprocessed_df[col] = interpolate_window(preprocessed_df, col, window=5)
                st.markdown("**Preprocessed Data Preview (NaNs filled)**")
                st.dataframe(preprocessed_df.style.format("{:.2f}"))
                if st.button("Download Preprocessed Data", key="download_preprocessed"):
                    with st.spinner("Generating preprocessed file..."):
                        output_buffer = BytesIO()
                        preprocessed_df.to_excel(output_buffer, index=False)
                        output_buffer.seek(0)
                        st.markdown(f'<div class="success-box">Preprocessing complete!</div>', unsafe_allow_html=True)
                        st.download_button(
                            label="Download Preprocessed Excel",
                            data=output_buffer,
                            file_name="preprocessed_pavement_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

process_tab(tab1, "INDEX_PCR", input_features["INDEX_PCR"], pcr_models)
process_tab(tab2, "RUT_AVG", input_features["RUT_AVG"], rut_models)

# Footer
st.markdown("---")
st.markdown("""
    ### How It Works
    - **Predict PCR/Rut**: Use sliders to input pavement data and predict PCR or Rut Average.
    - **Excel Predictor**: Upload an Excel file to predict values in bulk and download results.
    - **Visualize Data**: Upload an Excel file to compare actual vs. predicted values in line or scatter plots.
    - **Preprocess Data**: Upload an Excel file to fill missing values using interpolation.
    <br>
    *Built with ❤️ by the Pavement Analysis Team*
""", unsafe_allow_html=True)