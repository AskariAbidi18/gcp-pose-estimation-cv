import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.gcp_dataset import GCPDataset
from dataset.transforms import get_train_transforms
from models.gcp_model import GCPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = GCPDataset(
    "/content/GCP_Assignment_Datasets/train_dataset",
    transform=get_train_transforms(512)
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

model = GCPModel().to(device)

keypoint_loss = nn.MSELoss()
shape_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 20

best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, keypoints, labels in pbar:

        images = images.to(device)
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        pred_kp, pred_shape = model(images)

        loss_kp = keypoint_loss(pred_kp, keypoints)
        loss_shape = shape_loss(pred_shape, labels)

        loss = loss_kp + loss_shape

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss {epoch_loss:.4f}")

    # save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "gcp_model.pth")
        print("Saved new best model")