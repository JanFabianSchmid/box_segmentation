import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from src.model import UNet
from src.oscd_dataset import OscdDataset


def parse_args():
    """Parse arguments for visualization script"""
    parser = argparse.ArgumentParser(description="Box segmentation visualization script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="./oscd/coco_carton/oneclass_carton/",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the weights to be tested",
        default="example_training/best_epoch.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output location for the results of the experiment",
        default="./output_visu",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset_path)

    eval_dataset = OscdDataset(dataset_path, split="eval", image_divisible_ny=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Load the model weights
    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.is_file():
        print(f"Loading model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Checkpoint file not found: {checkpoint_path}")
        return

    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(eval_dataloader):
            # Get the data
            image = sample["image"].to(device)
            segmentation = sample["segmentation"].to(device)
            # Forward pass
            pred = model(image)

            # Save input image, ground truth segmentation, and predicted segmentation to disk
            img_tensor = image[0].cpu()
            TF.to_pil_image(img_tensor).save(os.path.join(output_dir, f"{idx}_input.png"))

            gt_tensor = segmentation[0].cpu()
            TF.to_pil_image(gt_tensor).save(os.path.join(output_dir, f"{idx}_gt.png"))

            pred_tensor = torch.sigmoid(pred[0]).cpu()  # Apply sigmoid for binary mask
            TF.to_pil_image(pred_tensor).save(os.path.join(output_dir, f"{idx}_pred.png"))
            TF.to_pil_image((pred_tensor > 0.5).float()).save(os.path.join(output_dir, f"{idx}_pred_binarized.png"))


if __name__ == "__main__":
    main()
