from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.responses import JSONResponse
import numpy as np
import io
from model import model

app = FastAPI()

def process(image: Image.Image) -> np.ndarray:
    image = image.resize((64, 64)).convert("L")
    image_arr = np.array(image).flatten() / 255.0
    f1 = np.mean(image_arr)
    f2 = np.var(image_arr)
    
    return np.array([[f1, f2]])

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image_content = process(image)
    label = model.predict(image_content)
    
    return JSONResponse(content={"result": "cat" if label[0] == 0 else "dog"})