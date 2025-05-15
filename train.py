import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import UNet
from src.oscd_dataset import OscdDataset


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description="Box segmentation training script")
    parser.add_argument(
        "--number_of_epochs",
        type=int,
        help="Specify the number of epochs to train",
        default=10,
    )
    parser.add_argument(
        "--apply_augmentations",
        type=boolean_string,
        help="Specify whether augmentations should be applied during training",
        default="False",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="./oscd/coco_carton/oneclass_carton/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output location for the results of the experiment",
        default="./output_train",
    )
    args = parser.parse_args()
    return args


def boolean_string(input_string: str) -> bool:
    if input_string not in {"False", "false", "True", "true"}:
        raise ValueError("Not a valid boolean string")
    return input_string in ["True", "true"]


class SummaryWriterTB:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self._global_step = 0
        self._epoch = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    def set_epoch(self, epoch):
        self._epoch = epoch

    def update_global_step(self):
        self._global_step = self._global_step + 1

    def add_scalar(self, tag, scalar_value, step):
        self.writer.add_scalar(tag, scalar_value, step)


def train_for_one_epoch(model, dataloader, optimizer, criterion, device, summary_writer):
    model.train()
    with tqdm(dataloader, desc="Training", leave=False) as pbar:
        for sample in pbar:
            # Get the data
            image = sample["image"].to(device)
            segmentation = sample["segmentation"].to(device)
            # Forward pass
            pred = model(image)
            # Backward pass
            loss = criterion(pred, segmentation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            summary_writer.update_global_step()
            summary_writer.add_scalar("Training/loss", loss.item(), summary_writer.global_step)


def test(model, dataloader, criterion, device, mode, summary_writer=None):
    model.eval()
    loss_accumulated = 0
    with torch.no_grad():
        for sample in tqdm(dataloader, desc=mode, leave=False):
            # Get the data
            image = sample["image"].to(device)
            segmentation = sample["segmentation"].to(device)
            # Forward pass
            pred = model(image)
            # Compute loss
            loss = criterion(pred, segmentation)
            loss_accumulated += loss.item()

    loss_avg = loss_accumulated / len(dataloader)
    print(f"Average {mode} Loss: {loss_avg:.2f}")
    if summary_writer:
        summary_writer.add_scalar("Validation/loss", loss_avg, summary_writer.epoch)
    return loss_avg


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset_path)

    train_dataset = OscdDataset(
        dataset_path,
        split="train",
        image_divisible_ny=16,
        apply_augmentations=args.apply_augmentations,
    )
    val_dataset = OscdDataset(
        dataset_path,
        split="val",
        image_divisible_ny=16,
    )

    # Currently, we can only use batch size of 1, because images are of different sizes
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create a binary cross-entropy loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    summary_writer = SummaryWriterTB(args.output_dir)
    summary_writer.set_epoch(0)

    best_error = test(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        mode="Initial Validation",
        summary_writer=summary_writer,
    )
    for epoch in range(args.number_of_epochs):
        print(f"Epoch {epoch + 1}/{args.number_of_epochs}")
        summary_writer.set_epoch(epoch + 1)
        train_for_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            summary_writer=summary_writer,
        )
        current_error = test(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            mode="Validation",
            summary_writer=summary_writer,
        )
        # Save the model
        model_save_path = output_dir / "latest_epoch.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")
        if current_error < best_error:
            best_error = current_error
            best_model_save_path = output_dir / "best_epoch.pth"
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved as {best_model_save_path}")


if __name__ == "__main__":
    main()
