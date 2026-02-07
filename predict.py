import os
import pickle
import torch
import clip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2  # OpenCV for face detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models.pkl"
OUTPUT_DIR = "output"
CACHE_DIR = "cache_embeddings"

THRESHOLDS = {
    "status": 0.4,
    "looks": 0.4,
    "wealth": 0.4,
    "age": 0.4
}

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Load models
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    models = pickle.load(f)

def embed_image(path):
    """Embed image and cache embedding for speed."""
    base = os.path.basename(path)
    cache_file = os.path.join(CACHE_DIR, base + ".npy")
    
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    emb_np = emb.cpu().numpy()[0]
    np.save(cache_file, emb_np)
    return emb_np

def predict_image(image_path):
    """Predict all tasks and save labeled image with face rectangles and probabilities."""
    emb = embed_image(image_path).reshape(1, -1)
    output = {}

    for task, data in models.items():
        clf = data.get("classifier")
        labels = data.get("labels")
        if clf is None or labels is None:
            print(f"Skipping {task}, no trained model found.")
            continue

        probs = clf.predict_proba(emb)[0]
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        result = labels[best_idx] if best_prob >= THRESHOLDS.get(task, 0.5) else "unknown"

        output[task] = {
            "result": result,
            labels[0]: float(probs[0]),
            labels[1]: float(probs[1])
        }

    # Open image for labeling
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Draw predictions with probabilities
    y_offset = 10
    for task, pred in output.items():
        text = f"{task}: {pred['result']} ({pred[list(pred.keys())[1]]:.2f}, {pred[list(pred.keys())[2]]:.2f})"
        draw.text((10, y_offset), text, fill=(0,255,0), font=font)
        y_offset += 30

    # Detect faces and draw rectangles with status/looks labels
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.21, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label rectangle with status or looks if available
        face_label = []
        if "status" in output: face_label.append(output["status"]["result"])
        if "looks" in output: face_label.append(output["looks"]["result"])
        if "wealth" in output: face_label.append(output["wealth"]["result"])
        if "age" in output: face_label.append(output["age"]["result"])
        label_text = ", ".join(face_label)
        cv2.putText(cv_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Convert back to PIL and save with incremental numbering
    labeled_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
    next_index = len(existing_files) + 1
    output_path = os.path.join(OUTPUT_DIR, f"labeled_{next_index}.jpg")
    labeled_img.save(output_path)

    print(f"Labeled image saved to {output_path}")
    return output

if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")
    if not os.path.exists(image_path):
        print("Image path does not exist.")
    else:
        res = predict_image(image_path)
        print(res)
