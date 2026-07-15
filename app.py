import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_dataset.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    cols = joblib.load("models/columns.pkl")
    scores = joblib.load("models/scores.pkl")
    fi = joblib.load("models/feature_importance.pkl")
    best_name = joblib.load("models/best_name.pkl")
    return model, scaler, cols, scores, fi, best_name

df = load_data()
model, scaler, cols, scores, fi, best_name = load_model()

st.title("Heart Disease Prediction")
st.write("Trains 5 models on patient data, tunes the best one, and predicts using it here.")

page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Model Comparison", "Predict"])

if page == "Home":
    st.header("Home")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", df.shape[0])
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Best Model", best_name)
    st.write("Note: the original CSV had 1025 rows but 723 of them were duplicates of the same ~300 patients, so those got dropped before training.")
    st.write("Use the sidebar to check out the data, look at some charts, compare models, or make a prediction.")

elif page == "Dataset":
    st.header("Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Summary stats")
    st.dataframe(df.describe(), use_container_width=True)
    st.write("Target distribution")
    st.bar_chart(df["target"].value_counts())

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Feature importance (random forest)")
        fi_s = pd.Series(fi).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7,6))
        fi_s.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.write("Age distribution by target")
    fig, ax = plt.subplots(figsize=(9,4))
    sns.histplot(data=df, x="age", hue="target", multiple="stack", ax=ax)
    st.pyplot(fig)

elif page == "Model Comparison":
    st.header("Model Comparison")
    scores_df = pd.DataFrame(scores).T
    st.dataframe(scores_df.style.format("{:.3f}"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9,4))
    scores_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0,1)
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.write("Best model selected: " + best_name)

elif page == "Predict":
    st.header("Make a Prediction")

    with st.form("form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
            trestbps = st.number_input("Resting Blood Pressure", 80, 220, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)

        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)

        with c3:
            slope = st.selectbox("Slope (0-2)", [0,1,2])
            ca = st.selectbox("Number of Major Vessels (0-4)", [0,1,2,3,4])
            thal = st.selectbox("Thal (0-3)", [0,1,2,3])

        submit = st.form_submit_button("Predict")

    if submit:
        row = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg,
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        input_df = pd.DataFrame([row])[cols]
        input_sc = scaler.transform(input_df)

        pred = model.predict(input_sc)[0]
        prob = model.predict_proba(input_sc)[0]

        st.divider()
        if pred == 1:
            st.error("Result: high chance of heart disease")
        else:
            st.success("Result: low chance of heart disease")

        st.write("Confidence - disease: " + str(round(prob[1]*100,1)) + "%, no disease: " + str(round(prob[0]*100,1)) + "%")

        result_df = input_df.copy()
        result_df["prediction"] = "disease" if pred == 1 else "no disease"
        result_df["disease_probability"] = round(prob[1], 3)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download this result as CSV", csv, "prediction_result.csv", "text/csv")
