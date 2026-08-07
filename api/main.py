from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image
from datetime import datetime
from pathlib import Path

from .disease_info import DISEASE_INFO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = (256, 256)

# Absolute path to model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "1.keras"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


@app.get("/ping")
async def ping():
    return {
        "message": "API is running"
    }


def read_file_as_image(data: bytes):
    """
    Reads uploaded image safely.
    Converts to RGB.
    Resizes to model input size.
    Returns numpy array.
    """

    image = Image.open(BytesIO(data)).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image = read_file_as_image(await file.read())

        img_batch = np.expand_dims(image, axis=0)

        predictions = model.predict(img_batch, verbose=0)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = round(float(np.max(predictions[0]) * 100), 2)

        disease = DISEASE_INFO.get(
            predicted_class,
            {
                "name": predicted_class,
                "severity": "Unknown",
                "description": "No information available.",
                "symptoms": [],
                "treatment": [],
                "prevention": [],
            },
        )

        # Confidence for every class
        all_predictions = []

        for i, class_name in enumerate(class_names):

            disease_name = DISEASE_INFO.get(
                class_name,
                {
                    "name": class_name
                }
            )["name"]

            all_predictions.append(
                {
                    "class": disease_name,
                    "confidence": round(float(predictions[0][i] * 100), 2)
                }
            )

        # Highest confidence first
        all_predictions = sorted(
            all_predictions,
            key=lambda x: x["confidence"],
            reverse=True
        )

        return {
            "class": disease["name"],
            "confidence": confidence,
            "severity": disease["severity"],
            "description": disease["description"],
            "symptoms": disease["symptoms"],
            "treatment": disease["treatment"],
            "prevention": disease["prevention"],

            # Extra Information
            "image_name": file.filename,
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "all_predictions": all_predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )