"""
Crop Clinic — FastAPI backend
Serves per-species plant disease classification models and
proxies treatment advice from the OpenAI API.
"""

import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import io

# ── Fix for Keras version mismatch (BatchNormalization renorm params) ──
import keras.src.layers.normalization.batch_normalization as _bn
_orig_init = _bn.BatchNormalization.__init__

def _patched_init(self, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _orig_init(self, **kwargs)

_bn.BatchNormalization.__init__ = _patched_init
# ── End fix ──

from dotenv import load_dotenv
load_dotenv()

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

app = FastAPI(title="Crop Clinic", version="1.0")

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
# Request model for treatment
# -----------------------

class TreatmentRequest(BaseModel):
    crop: str
    disease: str

# -----------------------
# Routes
# -----------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), media_type="text/html; charset=utf-8")

@app.get("/api/species")
async def get_species():
    """Return the list of available species for the dropdown."""
    species_list = []
    for species_name, cfg in SPECIES_CONFIG.items():
        species_list.append({
            "name": species_name,
            "num_classes": cfg["num_classes"],
            "single_class": cfg["single_class"],
        })
    return {"species": species_list}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), species: str = Form(...)):
    """Classify a leaf image using the model for the selected species."""

    if species not in SPECIES_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown species: {species}")

    cfg = SPECIES_CONFIG[species]
    class_names = cfg["class_names"]

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = preprocess_image(image)

        # Single-class species — no model needed
        if cfg["single_class"]:
            predicted_class = class_names[0]
            crop, disease = parse_prediction(predicted_class)
            return {
                "predicted_class": predicted_class,
                "confidence": 1.0,
                "crop": crop,
                "disease": disease,
                "prediction_text": f"{crop} — {disease} (100.0%)",
                "top_3": [{"class": predicted_class, "confidence": 1.0}],
            }

        # Multi-class species — run model
        if species not in LOADED_MODELS:
            raise HTTPException(status_code=500, detail=f"Model not loaded for {species}")

        model = LOADED_MODELS[species]
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

        return {
            "predicted_class": predicted_class,
            "confidence": top_confidence,
            "crop": crop,
            "disease": disease,
            "prediction_text": f"{crop} — {disease} ({top_confidence * 100:.1f}%)",
            "top_3": top_3,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/treatment")
async def treatment(req: TreatmentRequest):
    """Get treatment steps from OpenAI for the detected disease."""

    if req.disease.lower() == "healthy":
        return {
            "treatment": f"Your {req.crop} plant looks healthy! No treatment needed. "
                         f"Continue with regular watering, appropriate fertilisation, "
                         f"and monitoring for early signs of pests or disease."
        }

    try:
        import openai

        client = openai.OpenAI()  # reads OPENAI_API_KEY from env

        prompt = (
    		f"You are an agricultural expert. A farmer has a {req.crop} plant "
    		f"diagnosed with {req.disease}. Provide clear, actionable treatment steps. "
    		f"Include: 1) Immediate actions, 2) Chemical/organic treatment options, "
    		f"3) Prevention tips for the future. Keep it practical and concise. "
    		f"Do not use markdown formatting. Write in plain text only. "
    		f"Do not include any summary, conclusion, or closing statement at the end. "
    		f"Just list the steps and stop."
	)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )

        treatment_text = response.choices[0].message.content
        return {"treatment": treatment_text}

    except ImportError:
        raise HTTPException(status_code=500, detail="OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
