from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.oscd_dataset import OscdDataset


def train_for_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for idx, sample in enumerate(dataloader):
        # Get the data
        image = sample["image"].to(device)
        segmentation = sample["segmentation"].to(device)
        # Forward pass
        output = model(image)
        # Backward pass
        loss = criterion(output, segmentation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("loss: ", loss.item())


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            # Get the data
            image = sample["image"].to(device)
            segmentation = sample["segmentation"].to(device)
            # Forward pass
            output = model(image)
            # Compute loss
            loss = criterion(output, segmentation)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    print(f"Validation Loss: {val_loss}")


def main():
    number_of_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("/home/sjh3hi/box_segmentation/oscd/coco_carton/oneclass_carton/")
    output_folder = Path("output")
    output_folder.mkdir(parents=True, exist_ok=True)

    train_dataset = OscdDataset(dataset_path, split="train", image_divisible_ny=16)
    val_dataset = OscdDataset(dataset_path, split="val", image_divisible_ny=16)

    # Currently, we can only use batch size of 1, because images are of different sizes
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create a binary cross-entropy loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Initial validation")
    validate(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
    )
    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch + 1}/{number_of_epochs}")
        train_for_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )
        # Save the model
        model_save_path = output_folder / f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")


if __name__ == "__main__":
    main()
