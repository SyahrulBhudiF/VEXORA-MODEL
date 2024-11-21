from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from tempfile import NamedTemporaryFile
import shutil

app = FastAPI()

# Load model pada startup
model = load_model('modeldl.h5')
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


def preprocess_image(img_path, target_size=(128, 128)):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Gambar tidak ditemukan atau tidak dapat dimuat.")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            raise ValueError("Wajah tidak terdeteksi dalam gambar.")

        x, y, w, h = faces[0]
        face = img[y:y + h, x:x + w]

        face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_LANCZOS4)

        face_resized = face_resized.astype('float32') / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))
        return face_resized
    except Exception as e:
        raise ValueError(f"Kesalahan dalam memproses gambar: {e}")


def predict_emotion(img_path, mdl):
    processed_img = preprocess_image(img_path)

    predictions = mdl.predict(processed_img)
    predicted_class = np.argmax(predictions)

    emotions = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
    return emotions.get(predicted_class, "Unknown")


@app.post("/mood-detection")
async def detect_mood(
        image: UploadFile = File(...),
):
    try:
        if not image.content_type in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="File harus berupa gambar (JPEG atau PNG)"
            )

        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_path = temp_file.name

        try:
            emotion = predict_emotion(temp_path, model)

            os.unlink(temp_path)

            return JSONResponse(
                content={
                    "success": True,
                    "data": emotion,
                }
            )

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "shouldNotify": True,
                "message": str(e),
                "data": None
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
