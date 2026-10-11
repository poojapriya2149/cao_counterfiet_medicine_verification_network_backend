
import json
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image
from torchvision import transforms


# Resolve model files relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "mobilenetv4.xml"
CLASSES_PATH = MODEL_DIR / "medicine_classes.json"

IMAGE_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model and labels once when this module is imported.
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    classes = json.load(file)

core = ov.Core()
model = core.compile_model(str(MODEL_PATH), "CPU")

input_layer = model.input(0)
output_layer = model.output(0)


def predict_medicine(image_path):
    """
    Predict the medicine name from a package image.

    Returns:
        str: Predicted medicine name.
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = preprocess(image)

    input_data = image_tensor.unsqueeze(0).numpy()

    result = model({input_layer: input_data})[output_layer]
    probabilities = np.asarray(result)[0]

    predicted_index = int(np.argmax(probabilities))

    return classes[predicted_index]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python medicine_predict.py "path_to_image"',
            file=sys.stderr
        )
        sys.exit(1)

    try:
        print(predict_medicine(sys.argv[1]))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

# cd "D:\M.Tech Fall Sem 2026-27\computer architecture\project"
# .\openvino_env\Scripts\Activate.ps1
# python -c "import openvino as ov; print(ov.Core().available_devices)"
# python medicine_ai_pipeline.py
# python medicine_predict.py "D:\M.Tech Fall Sem 2026-27\computer architecture\project\dataset\test\test_img.jpg"