import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.oscd_dataset import OscdDataset
from train import test


def parse_args():
    """Parse arguments for evaluation script"""
    parser = argparse.ArgumentParser(description="Box segmentation evaluation script")
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

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

    # Create a binary cross-entropy loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    test(
        model=model,
        dataloader=eval_dataloader,
        criterion=criterion,
        device=device,
        mode="Evaluation",
    )


if __name__ == "__main__":
    main()
