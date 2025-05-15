import json
import random

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def get_all_segmentation_labels(annotations):
    segmentation_labels = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in segmentation_labels:
            segmentation_labels[image_id] = []
        segmentation_labels[image_id].append(annotation["segmentation"][0])
    return segmentation_labels


class OscdDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        image_divisible_ny,
        apply_augmentations=False,
    ):
        self.split = split
        self.apply_augmentations = apply_augmentations
        self.image_divisible_ny = image_divisible_ny
        self.images_path = dataset_path / "images"
        info_path = dataset_path / "annotations"
        if self.split == "train":
            self.images_path = self.images_path / "train2017"
            info_path = info_path / "instances_train2017.json"
        elif self.split in ["val", "eval"]:
            self.images_path = self.images_path / "val2017"
            info_path = info_path / "instances_val2017.json"
        else:
            raise ValueError("split must be either 'train' or 'val'")

        with open(info_path, "r") as file:
            self.info = json.load(file)

        self.segmentation_labels = get_all_segmentation_labels(self.info["annotations"])

        # Filter out images without segmentation labels
        self.info["images"] = [image for image in self.info["images"] if image["id"] in self.segmentation_labels]

        if self.split == "val":
            # Use the first half of validation images for 'val'
            self.info["images"] = self.info["images"][: len(self.info["images"]) // 2]
        elif self.split == "eval":
            # Use the second half of validation images for 'eval'
            self.info["images"] = self.info["images"][len(self.info["images"]) // 2 :]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.info["images"])

    def __getitem__(self, index):
        image_info = self.info["images"][index]
        image_id = image_info["id"]
        image_path = self.images_path / image_info["file_name"]

        # Load the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.to_tensor(image)

        # Create the segmentation mask
        segmentation_labels = self.segmentation_labels[image_id]
        # Start with an empty mask
        mask = Image.new("L", image.size, 0)
        # Draw the segmentation labels onto the mask
        mask_draw = ImageDraw.Draw(mask)
        for segmentation in segmentation_labels:
            # Image values will be divided by 255, when converting to tensor
            mask_draw.polygon(segmentation, outline=255, fill=255)
        # Convert the mask to a tensor
        mask_tensor = self.to_tensor(mask)

        # Resize the image and mask to be divisible by image_divisible_ny
        height, width = image_tensor.shape[1:3]
        new_height = (height // self.image_divisible_ny) * self.image_divisible_ny
        new_width = (width // self.image_divisible_ny) * self.image_divisible_ny
        image_tensor = transforms.functional.resize(image_tensor, (new_height, new_width))
        mask_tensor = transforms.functional.resize(mask_tensor, (new_height, new_width))

        if self.apply_augmentations:
            # Random rotation
            angle = random.uniform(-5, 5)
            image_tensor = F.rotate(image_tensor, angle)
            mask_tensor = F.rotate(mask_tensor, angle)

            # Photometric augmentations
            if random.random() < 0.5:
                image_tensor = F.adjust_brightness(image_tensor, 1 + (random.uniform(-0.1, 0.1)))
            if random.random() < 0.5:
                image_tensor = F.adjust_contrast(image_tensor, 1 + (random.uniform(-0.1, 0.1)))
            if random.random() < 0.5:
                image_tensor = F.adjust_saturation(image_tensor, 1 + (random.uniform(-0.1, 0.1)))
            if random.random() < 0.5:
                image_tensor = F.adjust_hue(image_tensor, random.uniform(-0.05, 0.05))
            # Gaussian noise
            if random.random() < 0.5:
                stddev = random.uniform(0.01, 0.01)
                noise = torch.randn_like(image_tensor) * stddev
                image_tensor = image_tensor + noise
                image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
            # Blur
            if random.random() < 0.5:
                image_tensor = F.gaussian_blur(image_tensor, kernel_size=3)

        sample_dict = {
            "image": image_tensor,
            "segmentation": mask_tensor,
        }
        return sample_dict
