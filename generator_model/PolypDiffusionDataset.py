import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import TrainingConfig

config = TrainingConfig()

class PolypDiffusionDataset(Dataset):
    def __init__(self, image_dirs, csv_files, mask_dirs=None, transformations=False, dreambooth=False, keep_one_class=None):
        self.image_paths = []
        self.labels = []
        self.dreambooth = dreambooth
        self.mask_paths = []
        self.has_masks = mask_dirs is not None

        self.dic_label2idx = {'AD': 0, 'ASS': 1, 'HP': 2}

        for i, (img_dir, csv_file) in enumerate(zip(image_dirs, csv_files)):
            self.df = pd.read_csv(csv_file)

            if keep_one_class:
                self.df = self.df[self.df["cls"] == keep_one_class]
                self.dic_label2idx = {keep_one_class: 0}

            for _, row in self.df.iterrows():
                img_path = os.path.join(img_dir, f"{row['image_id']}.tif")
                self.image_paths.append(img_path)
                self.labels.append(self.dic_label2idx[row["cls"]])

                if self.has_masks:
                    mask_dir = mask_dirs[i]
                    mask_path = os.path.join(mask_dir, f"{row['image_id']}.tif")
                    self.mask_paths.append(mask_path)

        self.idx2label = {idx: label for label, idx in self.dic_label2idx.items()}

        if self.dreambooth:
            self.class_token_map = {
                0: "a photo of sks adenomatous polyp",
                1: "a photo of zbt sessile serrated polyp",
                2: "a photo of mjt hyperplastic polyp"
            }

        if transformations:
            self.transformations_list = ['resize', 'randomHorizontalFlip', 'normalize']
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.ToTensor()
            self.transformations_list = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.has_masks:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
            image = np.array(image) * np.expand_dims(mask, axis=-1)
            image = Image.fromarray(image)

        image = self.transform(image)

        if self.dreambooth:
            prompt = self.class_token_map[label]
            return image, prompt
        else:
            return image, label
