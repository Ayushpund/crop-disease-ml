import json
from io import BytesIO
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

IMAGE_SIZE = (224, 224)
MODEL_PATH = "crop_model/best_model.keras"
CLASS_FILE = "class_indices.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(CLASS_FILE, "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}


def preprocess(bytes_img):
    img = Image.open(BytesIO(bytes_img)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img).astype("float32") / 255.
    img = np.expand_dims(img, 0)
    return img


@app.get("/")
def root():
    return {"status": "Crop Disease API Running on Railway!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess(img_bytes)

    preds = model.predict(img)[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "prediction": class_indices[class_id],
        "confidence": confidence,
        "probabilities": {
            class_indices[int(i)]: float(preds[i])
            for i in np.argsort(preds)[::-1][:3]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
