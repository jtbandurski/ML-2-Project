from model import MobileNetV3, EfficientNet
import torch
from utils import CONFIG, DEVICE
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from trainer import Trainer
from datasets import CustomDataset
from argparse import ArgumentParser
import os
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--overfit_single_batch", type=bool, default=False)
    parser.add_argument("--save_results", type=bool, default=False)
    parser.add_argument("--save_plots", type=bool, default=False)
    parser.add_argument("--train_with_augmentations", type=bool, default=False)
    args = parser.parse_args()

    if args.train_with_augmentations:
        df = pd.read_csv("data/train_metadata_with_augmentations.csv")
        df = df.groupby("common_name").filter(lambda x: len(x) >= 10)
    else:
        df = pd.read_csv("data/train_metadata.csv")
        df = df.groupby("common_name").filter(lambda x: len(x) >= 5)

    model = MobileNetV3(num_classes=len(set(df["primary_label"])))
    full_dataset = CustomDataset(
        df, target_sample_rate=CONFIG["sample_rate"], max_time=CONFIG["max_time"]
    )
    train_dataset, eval_dataset = full_dataset.train_test_split(stratify=True)
    train_dataset.encode_labels()
    eval_dataset.encode_labels()

    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, model_name=args.model_name
    )
    trainer.initialize_metrics_dict()
    train_dataloader, eval_dataloader = trainer.prepare_dataloaders(train_dataset, eval_dataset)

    best_eval_f1 = 0
    if args.overfit_single_batch:
        train_mels = next(iter(train_dataloader))[0].to(DEVICE)
        train_labels = next(iter(train_dataloader))[1].to(DEVICE)
        eval_mels = next(iter(eval_dataloader))[0].to(DEVICE)
        eval_labels = next(iter(eval_dataloader))[1].to(DEVICE)

        single_batch = {
            "train_mels": train_mels,
            "train_labels": train_labels,
            "eval_mels": eval_mels,
            "eval_labels": eval_labels,
        }

        for epoch in range(CONFIG["epochs"]):
            trainer.overfit_single_batch(
                batch=single_batch,
                epoch=epoch,
            )
    else:
        os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
        for epoch in range(CONFIG["epochs"]):
            training_metrics = trainer.train(dataloader=train_dataloader, epoch=epoch)
            eval_metrics = trainer.eval(dataloader=eval_dataloader, epoch=epoch)
            trainer.update_metrics_state(epoch, training_metrics, eval_metrics)

            if eval_metrics["weighted_metrics"]["f1"] > best_eval_f1:
                print(
                    f"Evaluation F1 Improved - {np.round(best_eval_f1, 2)} ---> {np.round(eval_metrics['weighted_metrics']['f1'], 2)}"
                )
                torch.save(
                    model.state_dict(),
                    f"{CONFIG['MODELS_DIR']}/{'_'.join(trainer.model_name.split(' '))}.bin",
                )
                print(f"Saved model checkpoint")
                best_eval_f1 = eval_metrics["weighted_metrics"]["f1"]

        if args.save_results:
            trainer.save_results_to_csv()
        if args.save_plots:
            trainer.prepare_and_save_plots()
