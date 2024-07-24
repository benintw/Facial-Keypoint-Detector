# %%writefile train.py

import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet


from config import CONFIG
from data_augs import train_transforms, val_transforms
from split_data import manual_split_training_data
from dataset import FacialKeypointDataset
from utils import get_submission, get_rmse, save_checkpoint


def train_one_epoch(loader, model, optimizer, loss_fn, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        scores[targets == -1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses)/num_examples) ** 0.5}")


def main():

    manual_split_training_data()  # split training to train_4, val_4, train_15, val_15 .csv

    config = CONFIG()

    train_ds = FacialKeypointDataset(
        csv_file="train_4.csv", transform=train_transforms, train=True
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    val_ds = FacialKeypointDataset(
        csv_file="val_4.csv", transform=val_transforms, train=False
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    test_ds = FacialKeypointDataset(
        csv_file="test.csv",
        transform=val_transforms,
        train=False,
    )

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = torch.nn.Linear(1280, 30)
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # model_4 = EfficientNet.from_pretrained("efficientnet-b0")
    # model_4._fc = torch.nn.Linear(1280, 30)
    # model_4 = model_4.to(config.DEVICE)

    # model_15 = EfficientNet.from_pretrained("efficientnet-b0")
    # model_15._fc = torch.nn.Linear(1280, 30)
    # model_15 = model_15.to(config.DEVICE)

    # # get_submission(test_loader, test_ds, model_15, model_4)

    for epoch in range(config.NUM_EPOCHS):
        get_rmse(val_loader, model, loss_fn, config.DEVICE)
        train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE)

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
