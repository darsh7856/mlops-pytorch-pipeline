import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "/app/checkpoints/classifier_v1.pt",
)

app = FastAPI(
    title="CIFAR-10 PyTorch Classifier",
    version="1.0.0",
)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_loaded = False

inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2470, 0.2435, 0.2616],
        ),
    ]
)


def load_model():
    global model
    global model_loaded

    checkpoint_file = Path(CHECKPOINT_PATH)

    if not checkpoint_file.exists():
        model_loaded = False
        return

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    architecture = checkpoint.get("architecture", "cifar_cnn")
    num_classes = checkpoint.get("num_classes", 10)

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_loaded = True


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    if model_loaded:
        return {
            "status": "ok",
            "model_loaded": True,
        }

    raise HTTPException(
        status_code=503,
        detail="Model checkpoint is not loaded",
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be an image",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    tensor = inference_transform(pil_image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "class_id": predicted_index,
        "class_name": class_names[predicted_index],
        "probabilities": {
            name: round(float(probabilities[index]), 6)
            for index, name in enumerate(class_names)
        },
    }