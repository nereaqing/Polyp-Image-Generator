import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AugmentedPolypClassificationDataset(Dataset):
    def __init__(self, dirs, image_size, transformations=False, ad_vs_rest=False):
        self.image_paths = []
        self.labels = []

        if ad_vs_rest:
            self.dic_label2idx = {'AD': 0, 'REST': 1}
        else:
            self.dic_label2idx = {'AD': 0, 'ASS': 1, 'HP': 2}
        
        for image_dir, csv_file in dirs:
            if csv_file is not None:
                self.df = pd.read_csv(csv_file)

                for _, row in self.df.iterrows():
                    img_path = os.path.join(image_dir, f"{row['image_id']}.tif")
                    label = row["cls"]
                    if ad_vs_rest:
                        label = 'REST' if label != 'AD' else 'AD'
                    
                    self.image_paths.append(img_path)
                    self.labels.append(self.dic_label2idx[label])
                    
            else:
                label = self.extract_label_from_dir(image_dir)
                for file in os.listdir(image_dir):
                    if file.endswith('.png'):
                        img_path = os.path.join(image_dir, file)
                        self.image_paths.append(img_path)
                        self.labels.append(self.dic_label2idx[label])
                

        self.dic_idx2label = {idx: label for label, idx in self.dic_label2idx.items()}

        if transformations:
            self.transformations_list = ['resize', 'randomHorizontalFlip', 'normalize']
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
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

        image = self.transform(image)

        return image, label

    def extract_label_from_dir(self, image_dir):
        label = os.path.basename(image_dir)
        if self.dic_label2idx.get("REST") is not None and label != "AD":
            return "REST"
        return label
