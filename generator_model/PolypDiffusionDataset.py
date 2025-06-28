import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config_diffusion import TrainingConfig

config = TrainingConfig()

class PolypDiffusionDataset(Dataset):
    def __init__(self, image_dirs, csv_files, mask_dirs=None, transformations=False, keep_one_class=None):
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        self.has_masks = mask_dirs is not None

        self.dic_label2idx = {}
        label_counter = 0

        # Convert to list for unified handling
        keep_one_class = [keep_one_class] if isinstance(keep_one_class, str) else keep_one_class

        for i, (img_dir, csv_file) in enumerate(zip(image_dirs, csv_files)):
            df = pd.read_csv(csv_file)

            if keep_one_class is not None:
                df = df[df["cls"].isin(keep_one_class)]

                if len(keep_one_class) > 1:
                    primary_cls = keep_one_class[0]
                    df["cls"] = df["cls"].apply(lambda x: primary_cls if x == primary_cls else "REST")

            for cls_name in df["cls"].unique():
                if cls_name not in self.dic_label2idx:
                    self.dic_label2idx[cls_name] = label_counter
                    label_counter += 1

            for _, row in df.iterrows():
                img_path = os.path.join(img_dir, f"{row['image_id']}.tif")
                self.image_paths.append(img_path)
                self.labels.append(self.dic_label2idx[row["cls"]])

                if self.has_masks:
                    mask_dir = mask_dirs[i]
                    mask_path = os.path.join(mask_dir, f"{row['image_id']}.tif")
                    self.mask_paths.append(mask_path)

        self.idx2label = {idx: label for label, idx in self.dic_label2idx.items()}

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

        return image, label
