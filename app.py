from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory, url_for  # type: ignore
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf  # type: ignore

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
ARTIFACT_DIRS = {
    "images": BASE_DIR / "images",
    "outputs": BASE_DIR / "outputs",
}

DISEASE_META = {
    "diabetes": {
        "name": "Diabetes",
        "accuracy": 92.54,
        "endpoint": "diabetesPage",
        "model_file": "diabetes.pkl",
        "input_count": 8,
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "accuracy": 97.66,
        "endpoint": "cancerPage",
        "model_file": "breast_cancer.pkl",
        "input_count": 22,
    },
    "liver": {
        "name": "Liver Disease",
        "accuracy": 71.18,
        "endpoint": "liverPage",
        "model_file": "liver.pkl",
        "input_count": 10,
    },
    "pneumonia": {
        "name": "Pneumonia",
        "accuracy": 91.35,
        "endpoint": "pneumoniaPage",
        "model_file": "pneumonia.h5",
        "input_count": 1,
    },
}

TABULAR_DISEASES = {
    DISEASE_META["diabetes"]["input_count"]: "diabetes",
    DISEASE_META["breast_cancer"]["input_count"]: "breast_cancer",
    DISEASE_META["liver"]["input_count"]: "liver",
}

MODEL_LABELS = {
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
    "SVM": "Support Vector Machine",
    "KNN": "K-Nearest Neighbors",
    "XGBoost": "XGBoost",
    "RF": "Random Forest",
    "GBDT": "Gradient Boosting",
}


def build_comparison_section(key, name, summary, insight, accuracies, roc_scores, artifact):
    model_order = ("LR", "DT", "SVM", "KNN", "XGBoost", "RF", "GBDT")
    models = []

    for short, accuracy, roc_score in zip(model_order, accuracies, roc_scores):
        models.append(
            {
                "short": short,
                "name": MODEL_LABELS[short],
                "accuracy": round(accuracy, 2),
                "roc_score": round(roc_score, 2),
            }
        )

    ranked_models = sorted(models, key=lambda model: model["accuracy"], reverse=True)
    best_model = ranked_models[0]
    runner_up = ranked_models[1]

    for model in models:
        model["is_best"] = model["short"] == best_model["short"]

    return {
        "key": key,
        "name": name,
        "summary": summary,
        "insight": insight,
        "models": models,
        "best_model": best_model,
        "runner_up": runner_up,
        "average_accuracy": round(sum(accuracies) / len(accuracies), 2),
        "spread": round(best_model["accuracy"] - min(accuracies), 2),
        "lead_margin": round(best_model["accuracy"] - runner_up["accuracy"], 2),
        "artifact": artifact,
    }


MODEL_COMPARISON_SECTIONS = [
    build_comparison_section(
        key="diabetes",
        name="Diabetes",
        summary="Cross-validated notebook accuracy scores for the tabular models trained on diabetes screening data.",
        insight="Random Forest performs best for Diabetes and creates the widest accuracy gap in this benchmark.",
        accuracies=[88.16, 86.4, 84.21, 83.33, 86.84, 92.98, 89.04],
        roc_scores=[86.94, 84.81, 81.38, 80.98, 85.08, 88.85, 87.62],
        artifact={
            "folder": "images",
            "filename": "PE_diabetes.jpeg",
            "caption": "Notebook-generated comparison chart used during diabetes model selection.",
        },
    ),
    build_comparison_section(
        key="breast_cancer",
        name="Breast Cancer",
        summary="Cross-validated notebook accuracy scores for the feature-based breast cancer classification study.",
        insight="Support Vector Machine performs best for Breast Cancer, with Gradient Boosting close behind above 97%.",
        accuracies=[95.91, 91.81, 97.66, 93.57, 94.74, 96.49, 97.08],
        roc_scores=[95.11, 92.92, 97.16, 92.26, 94.51, 94.78, 96.69],
        artifact={
            "folder": "images",
            "filename": "PE_breast_cancer.jpeg",
            "caption": "Notebook-generated comparison chart used during breast cancer model selection.",
        },
    ),
    build_comparison_section(
        key="liver",
        name="Liver Disease",
        summary="Cross-validated notebook accuracy scores for the liver disease benchmark across classic ML models.",
        insight="Support Vector Machine leads the liver benchmark, although the top three models remain tightly grouped.",
        accuracies=[69.41, 68.82, 71.18, 62.94, 68.24, 70.59, 70.0],
        roc_scores=[57.26, 49.37, 50.0, 55.75, 57.65, 61.54, 62.12],
        artifact={
            "folder": "outputs",
            "filename": "PE_liver.jpeg",
            "caption": "Notebook-generated comparison chart used during liver disease model selection.",
        },
    ),
]

PNEUMONIA_DASHBOARD = {
    "name": "Pneumonia",
    "model_name": "VGG19",
    "display_name": "VGG19 (Frozen CNN)",
    "summary": "Pneumonia is presented separately because the workflow is image-based and evaluated with a deep learning pipeline instead of the tabular model study.",
    "insight": "VGG19 reaches 87.50% validation accuracy and 79.17% test accuracy in the notebook evaluation.",
    "validation_accuracy": 87.5,
    "test_accuracy": 79.17,
    "validation_loss": 0.4487,
    "test_loss": 0.4851,
    "graphs": [
        {
            "folder": "outputs",
            "filename": "pneumonia_training_frozencnn.jpeg",
            "caption": "Training vs validation accuracy graph from the frozen VGG19 training run.",
        },
        {
            "folder": "outputs",
            "filename": "PE_pneumonia2.jpeg",
            "caption": "Notebook-generated performance summary chart for the pneumonia model study.",
        },
    ],
}

PNEUMONIA_DASHBOARD["average_accuracy"] = round(
    (PNEUMONIA_DASHBOARD["validation_accuracy"] + PNEUMONIA_DASHBOARD["test_accuracy"]) / 2, 2
)

AVERAGE_ACCURACY_BY_DISEASE = {
    section["key"]: section["average_accuracy"]
    for section in MODEL_COMPARISON_SECTIONS
}
AVERAGE_ACCURACY_BY_DISEASE["pneumonia"] = PNEUMONIA_DASHBOARD["average_accuracy"]

for disease_key, average_accuracy in AVERAGE_ACCURACY_BY_DISEASE.items():
    DISEASE_META[disease_key]["accuracy"] = average_accuracy

DASHBOARD_HIGHLIGHTS = [
    {
        "disease": section["name"],
        "model": section["best_model"]["name"],
        "accuracy": section["average_accuracy"],
        "detail": f"{section['best_model']['short']} leads this benchmark.",
    }
    for section in MODEL_COMPARISON_SECTIONS
]

DASHBOARD_HIGHLIGHTS.append(
    {
        "disease": PNEUMONIA_DASHBOARD["name"],
        "model": PNEUMONIA_DASHBOARD["model_name"],
        "accuracy": PNEUMONIA_DASHBOARD["average_accuracy"],
        "detail": "Shown separately with validation and test accuracy.",
    }
)


@app.context_processor
def inject_disease_meta():
    return {"disease_meta": DISEASE_META}


@lru_cache(maxsize=None)
def load_pickle_model(filename):
    with open(MODEL_DIR / filename, "rb") as model_file:
        return pickle.load(model_file)


@lru_cache(maxsize=1)
def load_pneumonia_model():
    return tf.keras.models.load_model(MODEL_DIR / DISEASE_META["pneumonia"]["model_file"])


def get_risk_level(risk_score):
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Moderate"
    return "Low"


def normalize_risk_score(score):
    return round(float(np.clip(score, 0, 100)), 2)


def build_diabetes_features(feature_map):
    engineered_features = {
        "NewBMI_Obesity 1": 0,
        "NewBMI_Obesity 2": 0,
        "NewBMI_Obesity 3": 0,
        "NewBMI_Overweight": 0,
        "NewBMI_Underweight": 0,
        "NewInsulinScore_Normal": 0,
        "NewGlucose_Low": 0,
        "NewGlucose_Normal": 0,
        "NewGlucose_Overweight": 0,
        "NewGlucose_Secret": 0,
    }

    if feature_map["BMI"] <= 18.5:
        engineered_features["NewBMI_Underweight"] = 1
    elif 24.9 < feature_map["BMI"] <= 29.9:
        engineered_features["NewBMI_Overweight"] = 1
    elif 29.9 < feature_map["BMI"] <= 34.9:
        engineered_features["NewBMI_Obesity 1"] = 1
    elif 34.9 < feature_map["BMI"] <= 39.9:
        engineered_features["NewBMI_Obesity 2"] = 1
    elif feature_map["BMI"] > 39.9:
        engineered_features["NewBMI_Obesity 3"] = 1

    if 16 <= feature_map["Insulin"] <= 166:
        engineered_features["NewInsulinScore_Normal"] = 1

    if feature_map["Glucose"] <= 70:
        engineered_features["NewGlucose_Low"] = 1
    elif 70 < feature_map["Glucose"] <= 99:
        engineered_features["NewGlucose_Normal"] = 1
    elif 99 < feature_map["Glucose"] <= 126:
        engineered_features["NewGlucose_Overweight"] = 1
    elif feature_map["Glucose"] > 126:
        engineered_features["NewGlucose_Secret"] = 1

    enriched_features = dict(feature_map)
    enriched_features.update(engineered_features)
    return np.asarray(list(map(float, enriched_features.values())), dtype=float)


def get_positive_class_probability(model, feature_array):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_array.reshape(1, -1))[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        positive_index = classes.index(1) if 1 in classes else int(np.argmax(probabilities))
        return float(probabilities[positive_index])

    prediction = model.predict(feature_array.reshape(1, -1))[0]
    return float(prediction)


def run_tabular_prediction(values, feature_map):
    disease_key = TABULAR_DISEASES.get(len(values))
    if not disease_key:
        return {"pred": "Invalid"}

    meta = DISEASE_META[disease_key]
    model = load_pickle_model(meta["model_file"])
    feature_array = np.asarray(values, dtype=float)

    if disease_key == "diabetes":
        feature_array = build_diabetes_features(feature_map)

    prediction = int(model.predict(feature_array.reshape(1, -1))[0])
    risk_score = normalize_risk_score(get_positive_class_probability(model, feature_array) * 100)

    return {
        "pred": prediction,
        "disease_key": disease_key,
        "disease_name": meta["name"],
        "accuracy": meta["accuracy"],
        "risk_score": risk_score,
        "risk_level": get_risk_level(risk_score),
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboardPage():
    return render_template(
        "dashboard.html",
        comparison_sections=MODEL_COMPARISON_SECTIONS,
        dashboard_highlights=DASHBOARD_HIGHLIGHTS,
        pneumonia_section=PNEUMONIA_DASHBOARD,
    )


@app.route("/artifacts/<folder>/<path:filename>")
def artifact(folder, filename):
    directory = ARTIFACT_DIRS.get(folder)

    if directory is None:
        abort(404)

    return send_from_directory(directory, filename)


@app.route("/diabetes", methods=["GET", "POST"])
def diabetesPage():
    return render_template("diabetes.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancerPage():
    return render_template("breast_cancer.html")


@app.route("/liver", methods=["GET", "POST"])
def liverPage():
    return render_template("liver.html")


@app.route("/pneumonia", methods=["GET", "POST"])
def pneumoniaPage():
    return render_template("pneumonia.html")


@app.route("/predict", methods=["POST", "GET"])
def predictPage():
    if request.method != "POST":
        return render_template("home.html")

    try:
        to_predict_dict = request.form.to_dict()

        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = int(value)
            except ValueError:
                to_predict_dict[key] = float(value)

        to_predict_list = list(map(float, list(to_predict_dict.values())))
        result = run_tabular_prediction(to_predict_list, to_predict_dict)

    except Exception as error:
        print("Error:", error)
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    if result.get("pred") == "Invalid":
        return render_template("predict.html", pred="Invalid")

    return render_template(
        "predict.html",
        pred=result["pred"],
        disease_name=result["disease_name"],
        accuracy=result["accuracy"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        retry_url=url_for(DISEASE_META[result["disease_key"]]["endpoint"]),
    )


@app.route("/pneumoniapredict", methods=["POST", "GET"])
def pneumoniapredictPage():
    if request.method != "POST":
        return render_template("pneumonia.html")

    try:
        image_file = request.files.get("image")
        if not image_file or image_file.filename == "":
            raise ValueError("Missing image")

        UPLOAD_DIR.mkdir(exist_ok=True)
        image_path = UPLOAD_DIR / "image.jpg"

        image = Image.open(image_file).convert("RGB")
        image.save(image_path)

        image_array = tf.keras.utils.load_img(image_path, target_size=(128, 128))
        image_array = tf.keras.utils.img_to_array(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        model = load_pneumonia_model()
        probabilities = model.predict(image_array, verbose=0)[0]
        pred = int(np.argmax(probabilities))
        risk_score = normalize_risk_score(probabilities[1] * 100)

    except Exception as error:
        print("PNEUMONIA ERROR:", error)
        message = "Please upload an image"
        return render_template("pneumonia.html", message=message)

    return render_template(
        "pneumonia_predict.html",
        pred=pred,
        disease_name=DISEASE_META["pneumonia"]["name"],
        accuracy=DISEASE_META["pneumonia"]["accuracy"],
        risk_score=risk_score,
        risk_level=get_risk_level(risk_score),
    )


if __name__ == "__main__":
    app.run(debug=True)

