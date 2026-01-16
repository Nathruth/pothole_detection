# src/predict.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from torchvision import transforms

MODEL_DIR = "../models"
MODEL_PATH = f"{MODEL_DIR}/mobilenetv2_pothole.onnx"
DATA_PATH  = f"{MODEL_DIR}/mobilenetv2_pothole.onnx.data"

# Load metadata
with open(DATA_PATH) as f:
    metadata = json.load(f)

classes = metadata["classes"]
mean = metadata["mean"]
std  = metadata["std"]
input_size = metadata["input_size"]

# Preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(img).unsqueeze(0).numpy().astype(np.float32)

# Load ONNX model
ort_session = ort.InferenceSession(MODEL_PATH)

def predict(image_path):
    x = preprocess_image(image_path)
    outputs = ort_session.run(None, {"input": x})
    pred = np.argmax(outputs[0], axis=1)[0]
    return classes[pred]

# Example usage
if __name__ == "__main__":
    img_path = "../data/normal/13.jpg"
    print("Prediction:", predict(img_path))
