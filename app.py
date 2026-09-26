import streamlit as st
import joblib
import pandas as pd


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

logistic_model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
forest_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🫀 Heart Disease Prediction")

st.write(
    "Machine Learning Disease Prediction System"
)

st.info(
    "For educational and demonstration purposes only. "
    "This application is not a medical diagnostic tool."
)


# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.subheader("Model Selection")

model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Support Vector Machine (SVM)",
        "Random Forest"
    ]
)


# --------------------------------------------------
# Patient information
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.slider(
        "Age",
        29,
        77,
        55
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.slider(
        "Resting Blood Pressure (mmHg)",
        94,
        200,
        140
    )

    chol = st.slider(
        "Cholesterol (mg/dL)",
        126,
        564,
        250
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [
            "≤ 120 mg/dL",
            "> 120 mg/dL"
        ]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )


with col2:

    thalach = st.slider(
        "Maximum Heart Rate",
        71,
        202,
        150
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        0.0,
        6.2,
        1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels (CA)",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )


# --------------------------------------------------
# Convert user-friendly values to dataset values
# --------------------------------------------------

sex_value = 1 if sex == "Male" else 0

cp_value = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4
}[cp]

fbs_value = 1 if fbs == "> 120 mg/dL" else 0

restecg_value = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}[restecg]

exang_value = 1 if exang == "Yes" else 0

slope_value = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}[slope]

thal_value = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}[thal]


# --------------------------------------------------
# Create patient DataFrame
# --------------------------------------------------

new_patient = pd.DataFrame([{
    "age": age,
    "sex": sex_value,
    "cp": cp_value,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_value,
    "restecg": restecg_value,
    "thalach": thalach,
    "exang": exang_value,
    "oldpeak": oldpeak,
    "slope": slope_value,
    "ca": ca,
    "thal": thal_value
}])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict",
    use_container_width=True
):

    # Select model and prepare input

    if model_name == "Logistic Regression":

        selected_model = logistic_model
        input_data = scaler.transform(new_patient)

    elif model_name == "Support Vector Machine (SVM)":

        selected_model = svm_model
        input_data = scaler.transform(new_patient)

    else:

        selected_model = forest_model
        input_data = new_patient


    # Make prediction

    prediction = selected_model.predict(input_data)[0]

    probabilities = selected_model.predict_proba(input_data)[0]


    # Display result

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: Disease detected"
        )

    else:

        st.success(
            "✅ Model Prediction: No disease detected"
        )


    # Display probabilities

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "No Disease",
            f"{probabilities[0]:.2%}"
        )

    with col2:

        st.metric(
            "Disease",
            f"{probabilities[1]:.2%}"
        )


# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()

st.subheader("About the Models")

st.write(
    "**Logistic Regression:** A statistical classification "
    "model used as a strong baseline for binary prediction."
)

st.write(
    "**Support Vector Machine (SVM):** A machine learning "
    "algorithm that finds a decision boundary between classes."
)

st.write(
    "**Random Forest:** An ensemble of decision trees that "
    "can capture more complex patterns in the data."
)