# Heart Disease Prediction - Model Optimization

Week 4 internship task. Trains 5 ML models on a heart disease dataset,
compares them, tunes the best one with GridSearchCV, and serves it through
a Streamlit dashboard.

## Live demo
Add your deployed link here after deploying, e.g. `https://your-app.streamlit.app`

## A note on the dataset

The uploaded `heart.csv` has 1025 rows, but 723 of them turned out to be exact
duplicates - it's really the same ~302 patients (the classic UCI heart disease
set) repeated a few times. The cleaning step drops those duplicates before any
training happens, otherwise the model would just be memorizing repeated rows.

## What it does

1. Loads `heart.csv`, drops duplicates, checks for missing values.
2. Trains 5 models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN.
3. Compares them on accuracy, precision, recall, F1, ROC-AUC.
4. Tunes the best-performing model with GridSearchCV.
5. Saves the tuned model with joblib.
6. Streamlit dashboard: home page, dataset overview, EDA, model comparison,
   and a prediction form with CSV download.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.800 | 0.848 | 0.824 | 0.871 |
| Decision Tree | 0.803 | 0.818 | 0.818 | 0.818 | 0.802 |
| Random Forest | 0.754 | 0.765 | 0.788 | 0.776 | 0.859 |
| SVM | 0.770 | 0.771 | 0.818 | 0.794 | 0.842 |
| KNN | 0.787 | 0.778 | 0.848 | 0.812 | 0.838 |
| Logistic Regression (Tuned) | 0.803 | 0.784 | 0.879 | 0.829 | 0.881 |

Logistic Regression and Decision Tree were nearly tied before tuning; Logistic
Regression edged ahead on accuracy so that's the one that got tuned. Tuning
(searching over C) kept accuracy the same but pushed recall and ROC-AUC up -
a fairly typical result, not every metric moves in the same direction.

## Project structure

```
Week4-Advanced-ML-Optimization/
├── data/
│   ├── heart.csv
│   └── cleaned_dataset.csv
├── notebooks/
│   └── model_training.ipynb
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   ├── scores.pkl
│   ├── feature_importance.pkl
│   └── best_name.pkl
├── screenshots/
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/<your-username>/Week4-Advanced-ML-Optimization.git
cd Week4-Advanced-ML-Optimization
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Retraining

Open `notebooks/model_training.ipynb` in Jupyter or Colab and run all cells.
It checks for the dataset at `/content/heart.csv` first (Colab), and falls
back to `../data/heart.csv` (repo). Re-running it regenerates everything in
`models/`.

## Dataset

Heart disease dataset - 302 unique patients after removing duplicates, 13
features plus target (0 = no disease, 1 = disease). Features cover age, sex,
chest pain type, blood pressure, cholesterol, and other clinical measurements.

## Tech stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib

## Possible improvements

- Try RandomizedSearchCV for a bigger search space without the runtime cost.
- Add SHAP for model explainability.
- Let users upload their own CSV for batch predictions.
- Deploy on Streamlit Community Cloud.
