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
    def __init__(self, data, target_sample_rate=CONFIG["sample_rate"], max_time=5, image_transforms=None):
        self.data = data
        self.file_paths = data["filename"].values
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        filepath = "data/train_audio/" + self.file_paths[index]
        audio, sample_rate = torchaudio.load(filepath, format="ogg")
        audio = self.to_mono(audio)

        if sample_rate != self.target_sample_rate:
            resample = Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)

        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)

        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)

        mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_mels=CONFIG["n_mels"], n_fft=CONFIG["n_fft"]
        )
        mel = mel_spectogram(audio)
        label = torch.tensor(self.labels[index])

        # Convert to Image
        image = torch.stack([mel, mel, mel])

        # Normalize Image
        max_val = torch.abs(image).max()
        image = image / max_val

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

    @staticmethod
    def train_test_split(data, test_split_ratio=CONFIG["eval_split_ratio"], stratify=True):
        if stratify:
            train_data, eval_data = train_test_split(
                data, test_size=0.2, stratify=data[CONFIG["stratify_column"]]
            )
        else:
            train_data, eval_data = train_test_split(
                data,
                test_size=0.2,
            )

        return train_data, eval_data
