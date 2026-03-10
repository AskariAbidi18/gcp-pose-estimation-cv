import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size):

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),

            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),

        ],
        keypoint_params=A.KeypointParams(format="xy")
    )


def get_val_transforms(img_size):

    return A.Compose(
        [],
        keypoint_params=A.KeypointParams(format="xy")
    )