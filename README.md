# 🌿 Crop Clinic

Machine learning-driven crop disease classification and remediation.

Crop Clinic is a web application that identifies plant diseases from leaf images using per-species convolutional neural networks, and generates treatment recommendations using the OpenAI API.

## Features

- **14 crop species supported** — Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **38 disease classifications** across all species
- **Per-species CNN models** — a dedicated TensorFlow model for each crop, trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Top-3 predictions** with confidence bars
- **AI treatment recommendations** powered by OpenAI GPT-4o-mini
- **Single-class species handling** — Blueberry, Orange, Raspberry, Soybean, and Squash return their diagnosis instantly without a model

## Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **ML Framework:** TensorFlow / Keras (tf-nightly)
- **Frontend:** HTML, CSS, JavaScript (no frameworks)
- **Treatment API:** OpenAI GPT-4o-mini
- **Training:** Jupyter Notebook, scikit-learn, matplotlib

## Project Structure

```
crop-clinic/
├── app.py                          # FastAPI backend
├── requirements.txt                # Python dependencies
├── .env                            # OpenAI API key (not committed)
├── .gitignore
├── templates/
│   └── index.html                  # Frontend
├── models/
│   ├── species_config.json         # Master config for all species
│   ├── apple/
│   │   ├── model.keras
│   │   └── class_names.txt
│   ├── cherry/
│   ├── corn_maize/
│   ├── grape/
│   ├── peach/
│   ├── pepper_bell/
│   ├── potato/
│   ├── strawberry/
│   └── tomato/
├── crop_clinic_per_species.ipynb   # Model training notebook
└── data/                           # Dataset (not committed)
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/virajs-garage/crop-clinic.git
cd crop-clinic
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...your-key-here
```

For the training notebook, set up your [Kaggle API token](https://www.kaggle.com/settings):

```
# Windows
mkdir %USERPROFILE%\.kaggle
echo YOUR_TOKEN > %USERPROFILE%\.kaggle\access_token
```

### 5. Train models (if not included)

Open `crop_clinic_per_species.ipynb` in VS Code or Jupyter and run all cells. This downloads the dataset and trains one model per species into the `models/` directory.

### 6. Run the app

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## How It Works

1. User selects a crop species from the dropdown
2. User uploads a photo of a leaf
3. The server resizes the image to 256×256 and passes it as a raw uint8 array to the appropriate per-species CNN model
4. The model returns disease probabilities — the app displays the top prediction and confidence bars for the top 3
5. User can request AI-generated treatment steps, which are fetched from the OpenAI API

**Important:** The models have a `Rescaling(1/255)` layer built in. Images must be passed as raw uint8 (0–255) without any normalization in the application code. Dividing by 255 before inference causes double-normalization and breaks all predictions.

## Model Performance

| Species | Classes | Best Val Accuracy |
|---------|---------|-------------------|
| Apple | 4 | 95% |
| Cherry | 2 | 100% |
| Corn (maize) | 4 | 94% |
| Grape | 4 | 78% |
| Peach | 2 | 99% |
| Pepper (bell) | 2 | 100% |
| Potato | 3 | 90% |
| Strawberry | 2 | 97% |
| Tomato | 10 | 81% |

## Dataset

[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) by Vipoooool on Kaggle. Contains 70,295 training images and 17,572 validation images across 38 classes.

## License

This project was built as part of a Computer Science Independent Studies course.
