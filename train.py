import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    num_workers=2,
    pin_memory=True
)


model = GCPModel().to(device)


keypoint_loss = nn.MSELoss()
shape_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


EPOCHS = 10


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for i, (images, keypoints, labels) in enumerate(loader):

        if i % 10 == 0:
            print("batch", i)

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

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss:.4f}")


torch.save(model.state_dict(), "gcp_model.pth")