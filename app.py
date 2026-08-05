from flask import (
    Flask,
    render_template,
    request
)

import cv2
import pickle
import numpy as np
import os

from src.preprocess import extract_vein
from src.feature_extraction import extract_features
from src.class_names import CLASS_NAMES

# ==========================
# Flask App
# ==========================

app = Flask(__name__)

# ==========================
# Load Model
# ==========================

model = pickle.load(
    open(
        "models/leaf_classifier.pkl",
        "rb"
    )
)

# ==========================
# Home Page
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================
# Prediction
# ==========================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    file = request.files[
        "leaf_image"
    ]

    if file.filename == "":

        return "No file selected"

    # ======================
    # Save Uploaded Image
    # ======================

    os.makedirs(
        "static/uploads",
        exist_ok=True
    )

    image_path = os.path.join(
        "static/uploads",
        file.filename
    )

    file.save(
        image_path
    )

    # ======================
    # Read Original Image
    # ======================

    original_image = cv2.imread(
        image_path
    )

    # ======================
    # Extract Veins
    # ======================

    vein_image = extract_vein(
        image_path,
        save_steps=True
    )

    # ======================
    # Extract Features
    # ======================

    features = extract_features(
        original_image,
        vein_image
    )

    if features is None:

        return "Feature extraction failed"

    features = np.array(
        [features]
    )

    # ======================
    # Prediction
    # ======================

    prediction = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    confidence = float(
        np.max(
            probabilities
        ) * 100
    )

    predicted_class = int(
        prediction[0]
    )

    # ======================
    # Unknown Plant Check
    # ======================

    if confidence < 60:

        plant_name = (
            "Plant Not Found in Dataset"
        )

    else:

        plant_name = CLASS_NAMES.get(
            predicted_class,
            "Unknown Plant"
        )
        print("Predicted Class =", predicted_class)

    return render_template(
        "result.html",
        plant_name=plant_name,
        confidence=round(
            confidence,
            2
        ),
        uploaded_image=image_path
    )

# ==========================
# Run
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )