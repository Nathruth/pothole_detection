# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import os
import json
import onnx

# -------------------- Settings --------------------
DATA_DIR = "../data"
MODEL_DIR = "../models"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Data --------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- Model --------------------
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)  # 2 classes
mobilenet = mobilenet.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=LEARNING_RATE)

# -------------------- Training --------------------
history = []

for epoch in range(NUM_EPOCHS):
    mobilenet.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = mobilenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    mobilenet.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = mobilenet(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = (sum([p == l for p, l in zip(val_preds, val_labels)]) / len(val_labels))
    f1 = pd.Series(val_preds).eq(pd.Series(val_labels)).mean()  # simplified F1 placeholder
    history.append({"epoch": epoch + 1, "loss": running_loss / len(train_loader), "val_accuracy": acc, "val_f1": f1})

    print(f"Epoch {epoch + 1}: Loss {running_loss / len(train_loader):.4f}, Val Acc {acc:.3f}")

# -------------------- Save ONNX --------------------
mobilenet.eval()
dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
onnx_path = os.path.join(MODEL_DIR, "mobilenetv2_pothole.onnx")
torch.onnx.export(
    mobilenet, dummy_input, onnx_path,
    export_params=True, opset_version=11, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("ONNX model saved.")

# -------------------- Save metadata --------------------
metadata = {
    "classes": train_dataset.classes,
    "input_size": [3, 224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}
with open(os.path.join(MODEL_DIR, "mobilenetv2_pothole.onnx.data"), "w") as f:
    json.dump(metadata, f, indent=4)

# -------------------- Save experiment history --------------------
pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "experiment_history.csv"), index=False)
print("Training complete and all files saved.")
