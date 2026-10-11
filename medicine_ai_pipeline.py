
import os
import json
import time
import random
import warnings
from pathlib import Path

os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import timm
import openvino as ov

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ================ CONFIGURATION ================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "dataset" / (
    "Mobile-Captured Pharmaceutical Medication Packages"
)

OUTPUT_DIR = PROJECT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mobilenetv4_conv_small.e1200_r224_in1k"
MODEL_PREFIX = "mobilenetv4"

NUM_CLASSES = 50
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", TRAIN_DEVICE)
print("Dataset:", DATA_DIR)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )


# ================ SELECT 50 RANDOM CLASSES ================

full_dataset = datasets.ImageFolder(str(DATA_DIR))

class_counts = {
    name: 0 for name in full_dataset.classes
}

for _, label in full_dataset.samples:
    class_counts[full_dataset.classes[label]] += 1

usable_classes = [
    name for name in full_dataset.classes
    if class_counts[name] >= 2
]

if len(usable_classes) < NUM_CLASSES:
    raise RuntimeError(
        f"Only {len(usable_classes)} classes have at least "
        f"two images. Cannot select {NUM_CLASSES} classes."
    )

# Random selection, reproducible with SEED.
rng = random.Random(SEED)

selected_classes = sorted(
    rng.sample(usable_classes, NUM_CLASSES)
)

class_to_label = {
    name: index
    for index, name in enumerate(selected_classes)
}

print(f"\nSelected {len(selected_classes)} classes:")

for name in selected_classes:
    print(f"  {name}: {class_counts[name]} images")

# Group image paths by selected class.
by_class = {
    index: []
    for index in range(len(selected_classes))
}

for path, original_label in full_dataset.samples:
    name = full_dataset.classes[original_label]

    if name in class_to_label:
        label = class_to_label[name]
        by_class[label].append(path)

# Split each class separately to preserve representation.
train_items = []
val_items = []

for label, paths in by_class.items():
    random.shuffle(paths)

    split = max(1, int(len(paths) * 0.8))
    split = min(split, len(paths) - 1)

    train_items.extend(
        (path, label) for path in paths[:split]
    )

    val_items.extend(
        (path, label) for path in paths[split:]
    )

random.shuffle(train_items)
random.shuffle(val_items)

print("\nTraining images:", len(train_items))
print("Validation images:", len(val_items))

# Save labels in the exact output-index order.
with open(
    OUTPUT_DIR / "medicine_classes.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(selected_classes, f, indent=2)


# ================ IMAGE TRANSFORMS ================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class MedicineDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        return self.transform(image), label


train_loader = DataLoader(
    MedicineDataset(train_items, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    MedicineDataset(val_items, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ================ TRAIN MOBILENETV4 ================

print(f"\nLoading pretrained model: {MODEL_NAME}")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=len(selected_classes)
)

# Freeze the feature extractor.
for parameter in model.parameters():
    parameter.requires_grad = False

# Train only the final classification layer.
for parameter in model.get_classifier().parameters():
    parameter.requires_grad = True

model = model.to(TRAIN_DEVICE)

optimizer = torch.optim.Adam(
    model.get_classifier().parameters(),
    lr=LEARNING_RATE
)

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()

    # Keep frozen batch-normalization layers in evaluation mode.
    for module in model.modules():
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm
        ):
            module.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(TRAIN_DEVICE)
        labels = labels.to(TRAIN_DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Training accuracy: {100 * correct / total:.2f}%"
    )

model.eval()

torch.save(
    model.state_dict(),
    OUTPUT_DIR / f"{MODEL_PREFIX}.pth"
)


# ================ PYTORCH VALIDATION ================

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(TRAIN_DEVICE)
        labels = labels.to(TRAIN_DEVICE)

        outputs = model(images)

        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += labels.size(0)

validation_accuracy = (
    100 * correct / total if total else 0.0
)

print(
    f"\nPyTorch validation accuracy: "
    f"{validation_accuracy:.2f}%"
)


# ================ OPENVINO EXPORT ================

print("\nConverting MobileNetV4 to OpenVINO...")

example_input = torch.randn(
    1, 3, IMAGE_SIZE, IMAGE_SIZE,
    device=TRAIN_DEVICE
)

ov_model = ov.convert_model(
    model,
    example_input=example_input
)

xml_path = OUTPUT_DIR / f"{MODEL_PREFIX}.xml"

ov.save_model(ov_model, str(xml_path))

print("OpenVINO XML:", xml_path)
print(
    "OpenVINO BIN:",
    OUTPUT_DIR / f"{MODEL_PREFIX}.bin"
)


# ================ CPU / GPU BENCHMARK ================

core = ov.Core()
benchmark_results = []

for device in ["CPU", "GPU"]:

    if device not in core.available_devices:
        print(f"\n{device} unavailable; skipping.")
        continue

    try:
        compiled = core.compile_model(ov_model, device)

    except Exception as error:
        print(f"\n{device} compilation failed: {error}")
        continue

    input_port = compiled.input(0)
    output_port = compiled.output(0)

    # Warm-up runs are excluded from measurements.
    first_images, _ = next(iter(val_loader))
    sample = first_images[0:1].numpy()

    for _ in range(5):
        compiled({input_port: sample})

    timings = []
    correct = 0
    total = 0

    for images, labels in val_loader:
        batch = images.numpy()

        for i in range(len(labels)):
            sample = batch[i:i + 1]

            start = time.perf_counter()

            result = compiled(
                {input_port: sample}
            )[output_port]

            timings.append(
                (time.perf_counter() - start) * 1000
            )

            prediction = int(
                np.argmax(result, axis=1)[0]
            )

            correct += int(
                prediction == int(labels[i])
            )

            total += 1

    avg_latency = float(np.mean(timings))
    accuracy = 100 * correct / total if total else 0.0
    fps = 1000 / avg_latency if avg_latency > 0 else 0.0

    row = {
        "model": MODEL_PREFIX,
        "device": device,
        "validation_accuracy_percent": round(accuracy, 2),
        "average_latency_ms": round(avg_latency, 3),
        "single_image_fps": round(fps, 2),
        "validation_images": total
    }

    benchmark_results.append(row)

    print(f"\nDevice: {device}")
    print(f"Validation accuracy: {accuracy:.2f}%")
    print(f"Average latency: {avg_latency:.3f} ms")
    print(f"Single-image FPS: {fps:.2f}")

with open(
    OUTPUT_DIR / "benchmark_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(benchmark_results, f, indent=2)


# ================ COMPLETE ================

print("\n========== COMPLETE ==========")
print("Selected classes:", len(selected_classes))
print("Model files:", OUTPUT_DIR)
print(
    "Class labels:",
    OUTPUT_DIR / "medicine_classes.json"
)
print(
    "Benchmarks:",
    OUTPUT_DIR / "benchmark_results.json"
)


# cd "D:\M.Tech Fall Sem 2026-27\computer architecture\project"
# .\openvino_env\Scripts\Activate.ps1
# python -c "import openvino as ov; print(ov.Core().available_devices)"
# python medicine_ai_pipeline.py
#