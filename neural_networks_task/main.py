from model import MobileNetV3
import torch
from utils import CONFIG
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from trainer import Trainer
from datasets import BirdCLEFDataset
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--overfit_single_batch", type=bool, default=False)
    args = parser.parse_args()

    df = pd.read_csv("data/train_metadata.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV3().to(device)

    train_dataset = BirdCLEFDataset(df, target_sample_rate=CONFIG["sample_rate"], max_time=CONFIG["max_time"])
    eval_dataset = BirdCLEFDataset(df, target_sample_rate=CONFIG["sample_rate"], max_time=CONFIG["max_time"])

    train_dataset.encode_labels()
    eval_dataset.encode_labels()

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    best_valid_f1 = 0
    trainer = Trainer(optimizer, model, device)
    train_dataloader, eval_dataloader = trainer.prepare_dataloaders(train_dataset, eval_dataset)

    if args.overfit_single_batch:
        train_mels = next(iter(train_dataloader))[0].to(device)
        train_labels = next(iter(train_dataloader))[1].to(device)
        eval_mels = next(iter(eval_dataloader))[0].to(device)
        eval_labels = next(iter(eval_dataloader))[1].to(device)

        single_batch = {
            "train_mels": train_mels,
            "train_labels": train_labels,
            "eval_mels": eval_mels,
            "eval_labels": eval_labels,
        }

        for epoch in range(CONFIG["epochs"]):
            trainer.overfit_single_batch(
                batch=single_batch,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
    else:
        for epoch in range(CONFIG["epochs"]):
            train_loss = trainer.train(
                train_dataloader, optimizer, scheduler, device, epoch, train_one_batch=True
            )
            valid_loss, valid_f1 = trainer.eval(eval_dataloader, device, epoch, eval_one_batch=True)
            if valid_f1 > best_valid_f1:
                print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
                torch.save(model.state_dict(), f"./model_0.bin")
                print(f"Saved model checkpoint at ./model_0.bin")
                best_valid_f1 = valid_f1
