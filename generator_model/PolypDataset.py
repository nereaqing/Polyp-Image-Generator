import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import TrainingConfig

config = TrainingConfig()

class PolypDataset(Dataset):
    def __init__(self, image_dir, csv_file, mask_dir=None, transformations=False, one_vs_rest=False, dreambooth=False, keep_one_class=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.dreambooth = dreambooth

        self.df = pd.read_csv(csv_file)
        if keep_one_class == "AD":
            self.image_ids = self.df[self.df['cls'] == 'AD']['image_id'].tolist()
            self.dic_label2idx = {"AD": 0}
        elif keep_one_class == "ASS":
            self.image_ids = self.df[self.df['cls'] == 'ASS']['image_id'].tolist()
            self.dic_label2idx = {"ASS": 0}
        elif keep_one_class == "HP":
            self.image_ids = self.df[self.df['cls'] == 'HP']['image_id'].tolist()
            self.dic_label2idx = {"HP": 0}
        else:
            self.image_ids = self.df['image_id'].tolist()
            self.dic_label2idx = {'AD': 0, 'ASS': 1, 'HP': 1 if one_vs_rest else 2}
        self.idx2label = {idx: label for label, idx in self.dic_label2idx.items()}
        self.labels = self.df['cls'].map(self.dic_label2idx).tolist()

        if self.dreambooth:
            # Class to DreamBooth prompt token
            self.class_token_map = {
                0: "a photo of sks adenomatous polyp",
                1: "a photo of zbt sessile serrated polyp",
                2: "a photo of mjt hyperplastic polyp"
            }

        if transformations:
            self.transformations_list = ['resize', 'randomHorizontalFlip', 'normalize']
            # self.transform = transforms.Compose([
            #     transforms.Resize((224, 224), antialias=True),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                          std=[0.229, 0.224, 0.225])
            # ])
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
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path).convert("RGB")

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, f"{img_id}.tif")
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
            image = np.array(image) * np.expand_dims(mask, axis=-1)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.dreambooth:
            prompt = self.class_token_map[label]
            return image, prompt
        
        else:
            return image, label

    def visualize_image(self, image, mask=None, masked_image=None):
        if mask is not None and masked_image is not None:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image)
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Mask (Polyp Region)")
            ax[1].axis('off')
            ax[2].imshow(masked_image)
            ax[2].set_title("Masked Image (Polyp Extracted)")
            ax[2].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.imshow(image)
            ax.set_title("Original Image")
            ax.axis('off')
        plt.show()
