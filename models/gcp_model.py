import torch
import torch.nn as nn
import torchvision.models as models


class GCPModel(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        # remove final FC layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.fc = nn.Linear(512, 256)

        # outputs
        self.keypoint_head = nn.Linear(256, 2)
        self.shape_head = nn.Linear(256, 3)

    def forward(self, x):

        x = self.feature_extractor(x)

        x = torch.flatten(x, 1)

        x = torch.relu(self.fc(x))

        keypoints = self.keypoint_head(x)
        shape_logits = self.shape_head(x)

        return keypoints, shape_logits