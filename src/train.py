import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/app/configs/training_config.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(str(config_path))

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    ).to(device)

    train_loader, validation_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=training_config.get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / output_config["model_name"]
    patience = training_config["early_stopping_patience"]

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(validation_loss)

        metrics = {
            "event": "epoch_complete",
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_accuracy, 5),
            "val_loss": round(validation_loss, 5),
            "val_accuracy": round(validation_accuracy, 5),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "device": str(device),
        }

        print(json.dumps(metrics), flush=True)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": validation_loss,
                    "val_accuracy": validation_accuracy,
                    "architecture": model_config["architecture"],
                    "num_classes": model_config["num_classes"],
                },
                checkpoint_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": str(checkpoint_path),
                    }
                ),
                flush=True,
            )
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                        }
                    ),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": round(best_val_loss, 5),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()