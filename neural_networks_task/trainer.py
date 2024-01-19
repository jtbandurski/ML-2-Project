from utils import loss_fn, CONFIG
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from datasets import BirdCLEFDataset
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        optimizer,
        model,
        device,
    ):
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def train(self, dataloader, optimizer, scheduler, device, epoch):
        self.model.train()
        total_loss = 0
        labels_list = []
        preds_list = []
        loop = tqdm(dataloader, position=0)
        for i, (mels, labels) in enumerate(loop):
            mels = mels.to(device)
            labels = labels.to(device)
            outputs = self.model(mels)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            loop.set_description(f"Training epoch [{epoch+1}/{CONFIG['epochs']}]")
            loop.set_postfix(loss=loss.item())

        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
        return total_loss / len(dataloader), accuracy, f1

    def eval(self, dataloader, device, epoch):
        self.model.eval()
        total_loss = 0
        preds_list = []
        labels_list = []

        loop = tqdm(dataloader, position=0)
        with torch.no_grad():
            for mels, labels in tqdm(dataloader):
                mels = mels.to(device)
                labels = labels.to(device)
                outputs = self.model(mels)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                preds_list.extend(preds.view(-1).cpu().detach().numpy())
                labels_list.extend(labels.view(-1).cpu().detach().numpy())
                loop.set_description(f"Evaluation epoch [{epoch+1}/{CONFIG['epochs']}]")
                loop.set_postfix(val_loss=loss.item())

        accuracy = accuracy_score(labels_list, preds_list)
        f1 = f1_score(labels_list, preds_list, average="macro")
        return total_loss / len(dataloader), accuracy, f1

    def overfit_single_batch(self, batch, optimizer, epoch):
        self.model.train()
        train_mels = batch["train_mels"]
        eval_mels = batch["eval_mels"]
        train_labels = batch["train_labels"]
        eval_labels = batch["eval_labels"]

        train_outputs = self.model(train_mels)
        train_loss = loss_fn(train_outputs, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.model.eval()
        with torch.no_grad():
            eval_outputs = self.model(eval_mels)
            eval_loss = loss_fn(eval_outputs, eval_labels)

        print(f"Epoch {epoch+1} | Train loss: {train_loss.item()} | Eval loss {eval_loss.item()}")

    @staticmethod
    def prepare_dataloaders(train_dataset, eval_dataset):
        train_dataloader = DataLoader(
            train_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True
        )
        return train_dataloader, eval_dataloader
