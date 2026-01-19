# src/predict.py
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from pathlib import Path

# -------- Paths (Docker-safe) --------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_pothole.onnx"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

# -------- Load metadata --------
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

classes = metadata["classes"]
mean = metadata["mean"]
std = metadata["std"]
input_size = metadata["input_size"]

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((input_size[1], input_size[2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    return x.numpy().astype(np.float32)

# -------- Load ONNX model (CPU) --------
ort_session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

# -------- Prediction --------
def predict(image_path: str) -> str:
    x = preprocess_image(image_path)
    outputs = ort_session.run(None, {"input": x})
    pred_idx = int(np.argmax(outputs[0], axis=1)[0])
    return classes[pred_idx]

# -------- Local test --------
if __name__ == "__main__":
    test_img = BASE_DIR / "data" / "normal" / "13.jpg"
    print("Prediction:", predict(str(test_img)))
