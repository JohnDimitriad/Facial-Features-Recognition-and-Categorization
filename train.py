import os
import pickle
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
MODEL_OUT = "models.pkl"
CACHE_DIR = "cache_embeddings"

os.makedirs(CACHE_DIR, exist_ok=True)

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def embed_image(path):
    # Cache filename
    base = os.path.basename(path)
    cache_file = os.path.join(CACHE_DIR, base + ".npy")
    
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    # Compute embedding
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    emb_np = emb.cpu().numpy()[0]
    
    # Save to cache
    np.save(cache_file, emb_np)
    
    return emb_np

def load_task(task_name, label_map):
    X, y = [], []

    for label_name, label_id in label_map.items():
        folder = os.path.join(DATASET_DIR, task_name, label_name)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            X.append(embed_image(path))
            y.append(label_id)

    return np.array(X), np.array(y)

tasks = {
    "status": {
        "labels": {"geeked": 0, "locked in": 1}
    },
    "looks": {
        "labels": {"chopped": 0, "goated": 1}
    },
    "wealth": {
        "labels": {"broke": 0, "blinged up": 1}
    },
    "age": {
        "labels": {"unc": 0, "nephew": 1}
    }
}

models = {}

for task, config in tasks.items():
    print(f"Training {task} model...")
    X, y = load_task(task, config["labels"])
    print(f"  Samples: {len(X)}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    models[task] = {
        "classifier": clf,
        "labels": list(config["labels"].keys())
    }

with open(MODEL_OUT, "wb") as f:
    pickle.dump(models, f)

print("All models saved to", MODEL_OUT)
