from utils import loss_fn, CONFIG, DEVICE
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from datasets import BirdCLEFDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import precision_recall_fscore_support as metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        model_name,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name

    def train(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        labels_list = []
        preds_list = []
        loop = tqdm(dataloader, position=0)
        for i, (mels, labels) in enumerate(loop):
            mels = mels.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = self.model(mels)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            labels_list.extend(labels.view(-1).cpu().detach().numpy())
            preds_list.extend(preds.view(-1).cpu().detach().numpy())
            loop.set_description(f"Training epoch [{epoch+1}/{CONFIG['epochs']}]")
            loop.set_postfix(loss=loss.item())

        metrics = self.calculate_metrics(labels_list, preds_list)
        total_loss = total_loss / len(dataloader)
        metrics["loss"] = total_loss
        return metrics

    def eval(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        preds_list = []
        labels_list = []

        loop = tqdm(dataloader, position=0)
        with torch.no_grad():
            for mels, labels in tqdm(dataloader):
                mels = mels.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self.model(mels)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                preds_list.extend(preds.view(-1).cpu().detach().numpy())
                labels_list.extend(labels.view(-1).cpu().detach().numpy())
                loop.set_description(f"Evaluation epoch [{epoch+1}/{CONFIG['epochs']}]")
                loop.set_postfix(loss=loss.item())

        metrics = self.calculate_metrics(labels_list, preds_list)
        total_loss = total_loss / len(dataloader)
        metrics["loss"] = total_loss
        return metrics

    def overfit_single_batch(self, batch, epoch):
        self.model.train()
        train_mels = batch["train_mels"]
        eval_mels = batch["eval_mels"]
        train_labels = batch["train_labels"]
        eval_labels = batch["eval_labels"]

        train_outputs = self.model(train_mels)
        train_loss = loss_fn(train_outputs, train_labels)
        train_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.eval()
        with torch.no_grad():
            eval_outputs = self.model(eval_mels)
            eval_loss = loss_fn(eval_outputs, eval_labels)

        print(f"Epoch {epoch+1} | Train loss: {train_loss.item()} | Eval loss {eval_loss.item()}")

    def calculate_metrics(
        self,
        labels,
        predictions,
    ):
        accuracy = accuracy_score(labels, predictions)
        macro_precision, macro_recall, macro_f1, _ = metrics(labels, predictions, average="macro")
        weighted_precision, weighted_recall, weighted_f1, _ = metrics(
            labels, predictions, average="weighted"
        )

        return {
            "accuracy": accuracy,
            "macro_metrics": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
            "weighted_metrics": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1,
            },
        }

    def update_metrics_state(
        self,
        epoch,
        training_metrics,
        eval_metrics,
    ):
        self.metrics_dict["epoch"].append(epoch)
        for metric in ["loss", "accuracy"]:
            self.metrics_dict[f"train_{metric}"].append(training_metrics[metric])
            self.metrics_dict[f"eval_{metric}"].append(eval_metrics[metric])

        for metric_type in ["macro_metrics", "weighted_metrics"]:
            for key in ["precision", "recall", "f1"]:
                train_key = f"train_{metric_type.split('_')[0]}_{key}"
                eval_key = f"eval_{metric_type.split('_')[0]}_{key}"
                self.metrics_dict[train_key].append(training_metrics[metric_type][key])
                self.metrics_dict[eval_key].append(eval_metrics[metric_type][key])

    def initialize_metrics_dict(self) -> None:
        self.metrics_dict = {
            "epoch": [],
            "train_loss": [],
            "eval_loss": [],
            "train_accuracy": [],
            "train_macro_precision": [],
            "train_macro_recall": [],
            "train_macro_f1": [],
            "train_weighted_recall": [],
            "train_weighted_precision": [],
            "train_weighted_f1": [],
            "eval_accuracy": [],
            "eval_macro_precision": [],
            "eval_macro_recall": [],
            "eval_macro_f1": [],
            "eval_weighted_recall": [],
            "eval_weighted_precision": [],
            "eval_weighted_f1": [],
        }

    def prepare_and_save_plots(self):
        """Creates and saves plots in PLOTS_DIR"""
        sns.set(rc={"figure.figsize": (16, 9)})
        os.makedirs(CONFIG["PLOTS_DIR"], exist_ok=True)
        metrics_df = pd.DataFrame(self.metrics_dict)
        melted_results = pd.melt(
            metrics_df, id_vars=["epoch"], var_name="metric", value_name="value"
        )
        accuracies = melted_results[melted_results["metric"].str.contains("accuracy")]
        loss_metrics = melted_results[melted_results["metric"].str.contains("loss")]
        macro_metrics = melted_results[melted_results["metric"].str.contains("macro")]
        weighted_metrics = melted_results[melted_results["metric"].str.contains("weighted")]

        dfs_list = [
            ("Accuracy", accuracies),
            ("Loss Metric", loss_metrics),
            ("Macro Metrics", macro_metrics),
            ("Weighted Metrics", weighted_metrics),
        ]

        for name, df in dfs_list:
            plt.figure()
            plot = sns.lineplot(data=df, x="epoch", y="value", hue="metric", marker="o")
            title = f"{name} Plot for {self.model_name}"
            plot.set_title(title, fontsize=30)
            plot.set_xlabel("Epoch", fontsize=20)
            plot.set_ylabel("Value", fontsize=20)
            filename = f"{CONFIG['PLOTS_DIR']}/{title}.png"
            plt.savefig(filename)
        print(f"Metrics plots have been saved in {CONFIG['PLOTS_DIR']}")

    def save_results_to_csv(self):
        """Saves training results to csv format in RESULTS_DIR"""
        os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
        metrics_df = pd.DataFrame(self.metrics_dict)
        filename = f"{CONFIG['RESULTS_DIR']}/results_{self.model_name}.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"Results have been saved in {CONFIG['RESULTS_DIR']}")

    @staticmethod
    def prepare_dataloaders(train_dataset, eval_dataset):
        train_dataloader = DataLoader(
            train_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True
        )
        return train_dataloader, eval_dataloader
