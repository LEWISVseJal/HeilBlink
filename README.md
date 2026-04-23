# HeilBlink

HeilBlink is a Flask-based medical prediction web application that combines classical machine learning and deep learning models to provide disease screening workflows through a browser interface.

The application currently supports:

- Diabetes prediction from tabular clinical inputs
- Breast cancer prediction from tabular diagnostic inputs
- Liver disease prediction from tabular medical inputs
- Pneumonia prediction from chest X-ray image uploads
- A dashboard that presents model comparison results and performance highlights

## Tech Stack

- Python
- Flask
- NumPy
- Pillow
- TensorFlow / Keras
- Scikit-image
- OpenCV
- HTML, CSS, JavaScript

## Project Structure

```text
HeilBlink/
|-- app.py                 # Main Flask application
|-- requirements.txt       # Python dependencies
|-- Procfile               # Deployment entrypoint for Gunicorn
|-- models/                # Trained model files (.pkl, .h5)
|-- data/                  # Datasets used for training and experimentation
|-- notebooks/             # Jupyter notebooks for model development
|-- templates/             # HTML templates for pages and prediction results
|-- static/                # CSS, JS, and static assets
|-- uploads/               # Temporary uploaded images for pneumonia prediction
|-- outputs/               # Generated charts and performance visuals
|-- images/                # Supporting project images and artifacts
```

## Features

### 1. Multi-disease prediction

The app exposes separate workflows for four prediction tasks:

- `/diabetes`
- `/cancer`
- `/liver`
- `/pneumonia`

Tabular disease routes collect form inputs and send them to saved pickle models. The pneumonia route accepts an uploaded image and runs inference using a TensorFlow model.

### 2. Performance dashboard

The `/dashboard` route presents:

- disease-wise model comparison summaries
- benchmark accuracy and ROC-style metrics
- highlighted best-performing models
- visual artifacts generated during notebook analysis

### 3. Risk-oriented output

Prediction results include:

- predicted class
- disease name
- model accuracy summary
- normalized risk score
- qualitative risk level (`Low`, `Moderate`, or `High`)

## Machine Learning Models

The project uses the following saved models:

- `models/diabetes.pkl`
- `models/breast_cancer.pkl`
- `models/liver.pkl`
- `models/pneumonia.h5`

Tabular models are loaded with `pickle`, while the pneumonia model is loaded with TensorFlow/Keras.

## How It Works

### Tabular prediction flow

1. The user enters clinical values into a disease-specific form.
2. The app converts form values into numeric features.
3. The corresponding model is loaded from the `models/` directory.
4. A prediction and positive-class probability are generated.
5. The probability is converted into a risk score and displayed on the result page.

### Pneumonia prediction flow

1. The user uploads a chest X-ray image.
2. The image is saved into the `uploads/` directory.
3. The app resizes and converts the image to a TensorFlow-ready array.
4. The pneumonia model performs inference.
5. The predicted class and risk score are rendered on the output page.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd HeilBlink
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Flask app locally with:

```bash
python app.py
```

By default, the app starts in debug mode because `app.py` contains:

```python
app.run(debug=True)
```

Then open the local server in your browser, typically at:

```text
http://127.0.0.1:5000/
```

## Deployment

The project includes a `Procfile` with:

```text
web: gunicorn app:app
```

This indicates the app is prepared for deployment on platforms that support Gunicorn-based Python web services.

## Main Routes

- `/` - Home page
- `/dashboard` - Model performance dashboard
- `/diabetes` - Diabetes prediction page
- `/cancer` - Breast cancer prediction page
- `/liver` - Liver disease prediction page
- `/pneumonia` - Pneumonia upload and prediction page
- `/predict` - Result handler for tabular disease models
- `/pneumoniapredict` - Result handler for pneumonia image prediction

## Datasets and Notebooks

The `data/` folder contains datasets used during model development. The `notebooks/` folder contains Jupyter notebooks for experimentation, training, and performance analysis for the supported diseases.

These notebooks appear to be the source of the comparison charts and benchmark summaries displayed in the dashboard.

## Notes

- The application is intended for academic or demonstration use.
- Predictions from this app should not be treated as medical advice or a clinical diagnosis.
- Model accuracy values shown in the UI are derived from project benchmarking data included in the code and notebook outputs.

## Future Improvements

- Add input validation and clearer error messaging
- Improve model explainability for end users
- Add automated tests for routes and prediction flows
- Support configurable upload filenames instead of a fixed image name
- Expand deployment and environment setup documentation

## License

This repository includes an MIT License. See the `LICENSE` file for details.
