"""Dataset and transforms. Images are read from the preprocessed cache (preprocess.py)."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from config import CONCEPTS, IMAGENET_MEAN, IMAGENET_STD


def train_tf(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),  # mild: colour is diagnostic
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_tf(size):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, concept_repeat=1):
        df = df.reset_index(drop=True)
        if concept_repeat > 1:  # oversample the few concept-labelled images so the concept head sees them
            has_c = df[CONCEPTS[0]] >= 0
            df = pd.concat([df] + [df[has_c]] * (concept_repeat - 1), ignore_index=True)
        self.df, self.tf = df, transform
        self.c = df[CONCEPTS].to_numpy(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.cache_path).convert("RGB")
        age = float(r.age) if pd.notna(r.age) else float("nan")
        return dict(image=self.tf(img), y=int(r.y), z=int(r.z), age=torch.tensor(age, dtype=torch.float32),
                    sex=int(r.sex), site=int(r.site), concepts=torch.from_numpy(self.c[i]), index=i)
