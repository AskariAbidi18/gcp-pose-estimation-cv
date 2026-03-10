import os
import json
import cv2
import torch
from torch.utils.data import Dataset


class GCPDataset(Dataset):
    def __init__(self, root_dir, img_size=512, transform=None):

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
            "L-Shape": 2
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        rel_path = self.image_paths[idx]
        label = self.labels[rel_path]

        img_name = os.path.basename(rel_path)
        img_path = os.path.join("/content/train_resized", img_name)

        image = cv2.imread(img_path)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        x = label["mark"]["x"]
        y = label["mark"]["y"]

        shape = label.get("verified_shape") or label.get("shape") or "Cross"

        shape_label = self.shape_map[shape]

        image = cv2.resize(image, (self.img_size, self.img_size))

        scale_x = self.img_size / w
        scale_y = self.img_size / h

        x = x * scale_x
        y = y * scale_y

        if self.transform:
            augmented = self.transform(image=image, keypoints=[(x, y)])
            image = augmented["image"]
            x, y = augmented["keypoints"][0]

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        x = x / self.img_size
        y = y / self.img_size

        keypoint = torch.tensor([x, y]).float()
        shape_label = torch.tensor(shape_label).long()

        return image, keypoint, shape_label