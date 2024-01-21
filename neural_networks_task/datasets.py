import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from utils import CONFIG
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import librosa
import sys


class BirdCLEFDataset(Dataset):
    def __init__(
        self, data, target_sample_rate=CONFIG["sample_rate"], max_time=5, image_transforms=None
    ):
        self.data = data
        self.file_paths = data["filename_tensor"].values
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        filepath = "data/tensors/" + self.file_paths[index]
        print(filepath)
        mel_filepath = os.path.join(filepath)
        mel = torch.load(mel_filepath)
        max_val = torch.abs(mel).max()
        mel_normalized = mel / max_val if max_val > 0 else mel
        image = torch.stack([mel_normalized, mel_normalized, mel_normalized])
        label = torch.tensor(self.labels[index])
        return image, label

    def encode_labels(self):
        encoder = LabelEncoder()
        self.data["primary_label_encoded"] = encoder.fit_transform(self.data["primary_label"])
        self.labels = self.data["primary_label_encoded"].values

    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio

    def crop_audio(self, audio):
        return audio[: self.num_samples]

    def to_mono(self, audio):
        return torch.mean(audio, axis=0)

    def train_test_split(self, test_split_ratio=CONFIG["eval_split_ratio"], stratify=True):
        if stratify:
            train_data, eval_data = train_test_split(
                self.data,
                test_size=test_split_ratio,
                stratify=self.data[CONFIG["stratify_column"]],
                random_state=42,
            )
        else:
            train_data, eval_data = train_test_split(
                self.data, test_size=test_split_ratio, random_state=42
            )
        train_dataset = BirdCLEFDataset(
            data=train_data,
            target_sample_rate=CONFIG["sample_rate"],
            max_time=CONFIG["max_time"],
        )
        eval_dataset = BirdCLEFDataset(
            data=eval_data,
            target_sample_rate=CONFIG["sample_rate"],
            max_time=CONFIG["max_time"],
        )

        return train_dataset, eval_dataset
