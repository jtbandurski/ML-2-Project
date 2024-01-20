import torch.nn as nn
import torch

CONFIG = {
    "epochs": 30,
    "num_classes": 264,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "eval_split_ratio": 0.2,
    "stratify_column": "primary_label",
    "sample_rate": 32_000,
    "hop_length": 512,
    "max_time": 5,
    "n_mels": 224,
    "n_fft": 1024,
    "learning_rate": 1e-5,
    "PLOTS_DIR": "./plots/results",
    "RESULTS_DIR": "./results",
    "MODELS_DIR": "./saved_models",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
