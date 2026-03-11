"""
Crop Clinic — Flask backend
Serves per-species plant disease classification models and
proxies treatment advice from the OpenAI API.
"""

import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

# -----------------------
# Configuration
# -----------------------

IMG_SIZE = 256  # Must match training notebook
CONFIG_PATH = os.path.join("models", "species_config.json")

# -----------------------
# Load species config
# -----------------------

with open(CONFIG_PATH, "r") as f:
    SPECIES_CONFIG = json.load(f)

# -----------------------
# Load all models at startup
# -----------------------

LOADED_MODELS = {}

for species_name, cfg in SPECIES_CONFIG.items():
    if not cfg["single_class"]:
        model_path = cfg["model_path"]
        if os.path.exists(model_path):
            print(f"Loading model: {species_name} ({model_path})")
            LOADED_MODELS[species_name] = tf.keras.models.load_model(model_path)
        else:
            print(f"WARNING: Model not found for {species_name}: {model_path}")

print(f"\nLoaded {len(LOADED_MODELS)} models. "
      f"{sum(1 for c in SPECIES_CONFIG.values() if c['single_class'])} single-class species registered.")

# -----------------------
# App Setup
# -----------------------

app = Flask(__name__)

# -----------------------
# Image Preprocessing
# -----------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and convert to uint8 numpy array.

    CRITICAL: Do NOT divide by 255 here.
    The model has a Rescaling(1/255) layer built in.
    Dividing here would cause double-normalisation and
    collapse all predictions to a single class.
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)
    return img_array

# -----------------------
# Helper: parse crop & disease
# -----------------------

def parse_prediction(class_name: str):
    """Split 'Apple___Black_rot' into ('Apple', 'Black rot')."""
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, disease

# -----------------------
# Routes
# -----------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/species", methods=["GET"])
def get_species():
    """Return the list of available species for the dropdown."""
    species_list = []
    for species_name, cfg in SPECIES_CONFIG.items():
        species_list.append({
            "name": species_name,
            "num_classes": cfg["num_classes"],
            "single_class": cfg["single_class"],
        })
    return jsonify({"species": species_list})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify a leaf image using the model for the selected species."""

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    if "species" not in request.form:
        return jsonify({"error": "No species selected"}), 400

    file = request.files["file"]
    species_name = request.form["species"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if species_name not in SPECIES_CONFIG:
        return jsonify({"error": f"Unknown species: {species_name}"}), 400

    cfg = SPECIES_CONFIG[species_name]
    class_names = cfg["class_names"]

    try:
        image = Image.open(file.stream)
        img_array = preprocess_image(image)

        # Single-class species — no model needed
        if cfg["single_class"]:
            predicted_class = class_names[0]
            crop, disease = parse_prediction(predicted_class)
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": 1.0,
                "crop": crop,
                "disease": disease,
                "prediction_text": f"{crop} — {disease} (100.0%)",
                "top_3": [{"class": predicted_class, "confidence": 1.0}],
            })

        # Multi-class species — run model
        if species_name not in LOADED_MODELS:
            return jsonify({"error": f"Model not loaded for {species_name}"}), 500

        model = LOADED_MODELS[species_name]
        predictions = model.predict(img_array, verbose=0)[0]

        top_index = int(np.argmax(predictions))
        top_confidence = float(predictions[top_index])

        # Top 3 predictions
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3 = [
            {
                "class": class_names[i],
                "confidence": float(predictions[i]),
            }
            for i in top_3_indices
        ]

        predicted_class = class_names[top_index]
        crop, disease = parse_prediction(predicted_class)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": top_confidence,
            "crop": crop,
            "disease": disease,
            "prediction_text": f"{crop} — {disease} ({top_confidence * 100:.1f}%)",
            "top_3": top_3,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/treatment", methods=["POST"])
def treatment():
    """Get treatment steps from OpenAI for the detected disease."""
    data = request.get_json()
    if not data or "crop" not in data or "disease" not in data:
        return jsonify({"error": "Missing crop or disease"}), 400

    crop = data["crop"]
    disease = data["disease"]

    if disease.lower() == "healthy":
        return jsonify({
            "treatment": f"Your {crop} plant looks healthy! No treatment needed. "
                         f"Continue with regular watering, appropriate fertilisation, "
                         f"and monitoring for early signs of pests or disease."
        })

    try:
        import openai

        client = openai.OpenAI()  # reads OPENAI_API_KEY from env

        prompt = (
            f"You are an agricultural expert. A farmer has a {crop} plant "
            f"diagnosed with {disease}. Provide clear, actionable treatment steps. "
            f"Include: 1) Immediate actions, 2) Chemical/organic treatment options, "
            f"3) Prevention tips for the future. Keep it practical and concise."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )

        treatment_text = response.choices[0].message.content
        return jsonify({"treatment": treatment_text})

    except ImportError:
        return jsonify({
            "error": "OpenAI package not installed. Run: pip install openai"
        }), 500
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500


# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    app.run(debug=True)
