import json

import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def get_all_segmentation_labels(annotations):
    segmentation_labels = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in segmentation_labels:
            segmentation_labels[image_id] = []
        segmentation_labels[image_id].append(annotation["segmentation"][0])
    return segmentation_labels


class OscdDataset(Dataset):
    def __init__(self, dataset_path, split, image_divisible_ny):
        self.image_divisible_ny = image_divisible_ny
        self.images_path = dataset_path / "images"
        info_path = dataset_path / "annotations"
        if split == "train":
            self.images_path = self.images_path / "train2017"
            info_path = info_path / "instances_train2017.json"
        elif split in ["val", "eval"]:
            self.images_path = self.images_path / "val2017"
            info_path = info_path / "instances_val2017.json"
        else:
            raise ValueError("split must be either 'train' or 'val'")

        with open(info_path, "r") as file:
            self.info = json.load(file)

        self.segmentation_labels = get_all_segmentation_labels(self.info["annotations"])

        # Filter out images without segmentation labels
        self.info["images"] = [image for image in self.info["images"] if image["id"] in self.segmentation_labels]

        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.info["images"])

    def __getitem__(self, index):
        image_info = self.info["images"][index]
        image_id = image_info["id"]
        image_path = self.images_path / image_info["file_name"]

        # Load the image
        image = Image.open(image_path).convert("RGB")
        # Normalize the image
        image_tensor = self.image_transform(image)

        # Create the segmentation mask
        segmentation_labels = self.segmentation_labels[image_id]
        # Start with an empty mask
        mask = Image.new("L", image.size, 0)
        # Draw the segmentation labels onto the mask
        mask_draw = ImageDraw.Draw(mask)
        for segmentation in segmentation_labels:
            mask_draw.polygon(segmentation, outline=1, fill=1)
        # Convert the mask to a tensor
        mask_tensor = self.mask_transform(mask)

        # Resize the image and mask to be divisible by image_divisible_ny
        height, width = image_tensor.shape[1:3]
        new_height = (height // self.image_divisible_ny) * self.image_divisible_ny
        new_width = (width // self.image_divisible_ny) * self.image_divisible_ny
        image_tensor = transforms.functional.resize(image_tensor, (new_height, new_width))
        mask_tensor = transforms.functional.resize(mask_tensor, (new_height, new_width))

        sample_dict = {
            "image": image_tensor,
            "segmentation": mask_tensor,
        }
        return sample_dict
