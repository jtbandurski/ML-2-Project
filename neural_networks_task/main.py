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
    df = df.groupby("common_name").filter(lambda x: len(x) >= 5)
    print(df.head())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV3().to(device)
    full_dataset = BirdCLEFDataset(
        df, target_sample_rate=CONFIG["sample_rate"], max_time=CONFIG["max_time"]
    )
    train_dataset, eval_dataset = full_dataset.train_test_split(stratify=True)

    train_dataset.encode_labels()
    eval_dataset.encode_labels()

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    trainer = Trainer(optimizer, model, device)
    train_dataloader, eval_dataloader = trainer.prepare_dataloaders(train_dataset, eval_dataset)

    best_eval_f1 = 0
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
                epoch=epoch,
            )
    else:
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_accuracy, train_f1 = trainer.train(
                train_dataloader, optimizer, scheduler, device, epoch
            )
            valid_loss, eval_accuracy, eval_f1 = trainer.eval(
                eval_dataloader, device, epoch, eval_one_batch=False
            )
            print(
                f"Statistics for epoch {epoch+1}: \n \
                Train Accuracy: {train_accuracy} \n \
                Train F1: {train_f1} \n \
                Eval Accuracy: {eval_accuracy} \n \
                Eval F1: {eval_f1}"
            )
            if eval_f1 > best_eval_f1:
                print(f"Validation F1 Improved - {best_eval_f1} ---> {eval_f1}")
                torch.save(model.state_dict(), f"./model_0.bin")
                print(f"Saved model checkpoint at ./model_0.bin")
                best_eval_f1 = eval_f1
