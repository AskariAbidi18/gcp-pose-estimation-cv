import os
import json
import cv2
from sqlalchemy import label
import torch
from torch.utils.data import Dataset


class GCPDataset(Dataset):
    def __init__(self, root_dir, img_size=512, transform=None):
        """
        root_dir: path to train_dataset
        img_size: resized image size
        transform: albumentations transforms
        """

        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform

        label_path = os.path.join(root_dir, "gcp_marks.json")

        with open(label_path) as f:
            self.labels = json.load(f)

        self.image_paths = list(self.labels.keys())

        self.shape_map = {
            "Cross": 0,
            "Square": 1,
            "L-Shaped": 2,
            "L-Shape" : 2
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        rel_path = self.image_paths[idx]
        label = self.labels[rel_path]

        img_path = os.path.join(self.root_dir, rel_path)

        image = cv2.imread(img_path)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        x = label["mark"]["x"]
        y = label["mark"]["y"]

        shape = label.get("verified_shape", None)

        if shape is None:
            shape = label.get("shape", None)

        if shape is None:
            shape = "Cross"   

        shape_label = self.shape_map[shape]

        # resize image
        image = cv2.resize(image, (self.img_size, self.img_size))

        # scale coordinates
        scale_x = self.img_size / w
        scale_y = self.img_size / h

        x = x * scale_x
        y = y * scale_y

        if self.transform:
            augmented = self.transform(image=image, keypoints=[(x, y)])
            image = augmented["image"]
            x, y = augmented["keypoints"][0]

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        keypoint = torch.tensor([x, y]).float()
        shape_label = torch.tensor(shape_label).long()

        return image, keypoint, shape_label