from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Machine Learning Assignment-2 : Dry Bean Classification",
    page_icon="",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 750;
            color: #16324F;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #52606D;
            margin-bottom: 1.5rem;
        }

        .info-box {
            padding: 1rem;
            border-radius: 0.7rem;
            background-color: #EEF6FF;
            border-left: 5px solid #2563EB;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 0.75rem;
            border-radius: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


# ---------------------------------------------------------
# Load resources
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Required model file was not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_default_test_data():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data file was not found: {TEST_DATA_PATH}"
        )

    return pd.read_csv(TEST_DATA_PATH)


try:
    models = load_models()
    metadata = load_metadata()
    default_test_data = load_default_test_data()

except Exception as error:
    st.error(f"Application resources could not be loaded: {error}")
    st.stop()


FEATURE_NAMES = metadata["feature_names"]
TARGET_COLUMN = metadata.get("target_column", "Class")
CLASS_NAMES = metadata["class_names"]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def validate_data(data):
    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in data.columns
    ]

    if missing_features:
        return False, (
            "The uploaded CSV is missing these required features: "
            + ", ".join(missing_features)
        )

    non_numeric_features = []

    for feature in FEATURE_NAMES:
        converted = pd.to_numeric(data[feature], errors="coerce")

        if converted.isna().all() and not data[feature].isna().all():
            non_numeric_features.append(feature)

    if non_numeric_features:
        return False, (
            "These features do not contain valid numeric values: "
            + ", ".join(non_numeric_features)
        )

    return True, ""


def prepare_features(data):
    features = data[FEATURE_NAMES].copy()

    for feature in FEATURE_NAMES:
        features[feature] = pd.to_numeric(
            features[feature],
            errors="coerce",
        )

    return features


def calculate_metrics(model, features, actual_values):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    model_classes = model.named_steps["classifier"].classes_

    metrics = {
        "Accuracy": accuracy_score(
            actual_values,
            predictions,
        ),
        "AUC": roc_auc_score(
            actual_values,
            probabilities,
            labels=model_classes,
            multi_class="ovr",
            average="macro",
        ),
        "Precision": precision_score(
            actual_values,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            actual_values,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1": f1_score(
            actual_values,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            actual_values,
            predictions,
        ),
    }

    return metrics, predictions, probabilities


def create_confusion_matrix(actual_values, predictions):
    matrix = confusion_matrix(
        actual_values,
        predictions,
        labels=CLASS_NAMES,
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return figure


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Machine Learning Assignment-2 : Dry Bean Classification Lab</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Compare five machine-learning classification models using
        morphological measurements of dry beans.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
        This application evaluates Logistic Regression, Decision Tree,
        K-Nearest Neighbors, Gaussian Naive Bayes and Random Forest.
        The dataset contains 16 numerical features and seven bean classes.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Experiment controls")

selected_model_name = st.sidebar.selectbox(
    "Select a classification model",
    options=list(MODEL_FILES.keys()),
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data",
    type=["csv"],
    help=(
        "Upload a CSV containing the 16 required feature columns. "
        "Include the Class column to calculate evaluation metrics."
    ),
)

use_default_data = st.sidebar.checkbox(
    "Use repository test data",
    value=uploaded_file is None,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AUC uses macro one-vs-rest averaging. Precision, recall and "
    "F1 use weighted averaging."
)


# ---------------------------------------------------------
# Select input data
# ---------------------------------------------------------

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        data_source = "Uploaded CSV"

    except Exception as error:
        st.error(f"The uploaded CSV could not be read: {error}")
        st.stop()

elif use_default_data:
    input_data = default_test_data.copy()
    data_source = "Repository test data"

else:
    st.info(
        "Upload a CSV file or select “Use repository test data” "
        "to begin."
    )
    st.stop()


is_valid, validation_message = validate_data(input_data)

if not is_valid:
    st.error(validation_message)
    st.stop()


# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------

st.subheader("1. Test data")

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.metric("Data source", data_source)
overview_col2.metric("Number of rows", f"{len(input_data):,}")
overview_col3.metric(
    "Actual labels available",
    "Yes" if TARGET_COLUMN in input_data.columns else "No",
)

with st.expander("Preview test data", expanded=False):
    st.dataframe(
        input_data.head(25),
        use_container_width=True,
    )

features = prepare_features(input_data)


# ---------------------------------------------------------
# Prediction section
# ---------------------------------------------------------

st.subheader("2. Selected model predictions")

selected_model = models[selected_model_name]
selected_predictions = selected_model.predict(features)

prediction_output = input_data.copy()
prediction_output["Predicted_Class"] = selected_predictions

if TARGET_COLUMN in input_data.columns:
    prediction_output["Correct_Prediction"] = (
        input_data[TARGET_COLUMN].astype(str)
        == pd.Series(selected_predictions).astype(str)
    )

st.write(f"Selected model: **{selected_model_name}**")

st.dataframe(
    prediction_output.head(25),
    use_container_width=True,
)

prediction_csv = prediction_output.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download predictions",
    data=prediction_csv,
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# Evaluation section
# ---------------------------------------------------------

if TARGET_COLUMN not in input_data.columns:
    st.warning(
        "Predictions were generated successfully. Add a Class column "
        "to the CSV to calculate accuracy, AUC, precision, recall, "
        "F1 and MCC."
    )
    st.stop()


actual_values = input_data[TARGET_COLUMN].astype(str).str.strip()

unknown_classes = sorted(
    set(actual_values.unique()) - set(CLASS_NAMES)
)

if unknown_classes:
    st.error(
        "The Class column contains unknown labels: "
        + ", ".join(unknown_classes)
    )
    st.stop()


# ---------------------------------------------------------
# Compare all five models
# ---------------------------------------------------------

st.subheader("3. Comparison of all models")

comparison_rows = []
evaluation_cache = {}

for model_name, model in models.items():
    try:
        metrics, predictions, probabilities = calculate_metrics(
            model,
            features,
            actual_values,
        )

    except ValueError as error:
        st.error(
            f"Metrics could not be calculated for {model_name}: {error}"
        )
        st.stop()

    evaluation_cache[model_name] = {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    comparison_rows.append({
        "ML Model Name": model_name,
        **metrics,
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    by="F1",
    ascending=False,
).reset_index(drop=True)

metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

st.dataframe(
    comparison_df.style.format(
        {column: "{:.4f}" for column in metric_columns}
    ).background_gradient(
        subset=metric_columns,
        cmap="Blues",
    ),
    use_container_width=True,
    hide_index=True,
)

winner_name = comparison_df.iloc[0]["ML Model Name"]

st.success(
    f"Overall winner based on weighted F1 score: {winner_name}"
)

chart_data = comparison_df.set_index("ML Model Name")[metric_columns]
st.bar_chart(chart_data)


# ---------------------------------------------------------
# Selected model metrics
# ---------------------------------------------------------

st.subheader(f"4. Detailed results of : {selected_model_name}")

selected_results = evaluation_cache[selected_model_name]
selected_metrics = selected_results["metrics"]
selected_predictions = selected_results["predictions"]

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col1.metric(
    "Accuracy",
    f"{selected_metrics['Accuracy']:.4f}",
)
metric_col2.metric(
    "AUC Score",
    f"{selected_metrics['AUC']:.4f}",
)
metric_col3.metric(
    "Precision",
    f"{selected_metrics['Precision']:.4f}",
)
metric_col4.metric(
    "Recall",
    f"{selected_metrics['Recall']:.4f}",
)
metric_col5.metric(
    "F1 Score",
    f"{selected_metrics['F1']:.4f}",
)
metric_col6.metric(
    "MCC Score",
    f"{selected_metrics['MCC']:.4f}",
)


# ---------------------------------------------------------
# Confusion matrix and classification report
# ---------------------------------------------------------

tab1, tab2 = st.tabs(
    ["Confusion Matrix", "Classification Report"]
)

with tab1:
    confusion_figure = create_confusion_matrix(
        actual_values,
        selected_predictions,
    )

    st.pyplot(confusion_figure)
    plt.close(confusion_figure)

with tab2:
    report = classification_report(
        actual_values,
        selected_predictions,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True,
    )


# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------

with st.expander("Evaluation methodology"):
    st.markdown(
        """
        - **Accuracy:** proportion of correctly Classified Test Records.
        - **AUC:** Macro-Average One-vs-Rest Multiclass AUC.
        - **Precision:** Weighted Average Precision.
        - **Recall:** Weighted Average Recall.
        - **F1:** Weighted Average F1-Score.
        - **MCC:** Multiclass Matthews Correlation Coefficient.
        - The overall winner is identified using Weighted F1-Score.
        """
    )
