import os
import json
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gcp_model import GCPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512

TEST_DIR = "/content/GCP_Assignment_Datasets/test_dataset"


shape_map_rev = {
    0: "Cross",
    1: "Square",
    2: "L-Shaped"
}


model = GCPModel().to(device)
model.load_state_dict(torch.load("gcp_model.pth", map_location=device))
model.eval()


predictions = {}


image_paths = []

for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            full_path = os.path.join(root, file)

            rel_path = os.path.relpath(full_path, TEST_DIR)

            image_paths.append((full_path, rel_path))


for full_path, rel_path in tqdm(image_paths):

    image = cv2.imread(full_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = torch.tensor(image).permute(2,0,1).float()/255.0

    image = image.unsqueeze(0).to(device)


    with torch.no_grad():

        kp_pred, shape_logits = model(image)


    kp_pred = kp_pred.squeeze().cpu().numpy()

    x = kp_pred[0] * w
    y = kp_pred[1] * h


    shape_idx = torch.argmax(shape_logits, dim=1).item()

    shape_name = shape_map_rev[shape_idx]


    predictions[rel_path] = {
        "mark": {
            "x": float(x),
            "y": float(y)
        },
        "verified_shape": shape_name
    }


with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)


print("predictions.json generated")