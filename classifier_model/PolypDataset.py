import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PolypDataset(Dataset):
    def __init__(self, image_dir, csv_file, mask_dir=None, transformations=False, one_vs_rest=False):
        self.img_paths = []
        self.labels = []
        self.images = []
        self.masks = []
        self.cropped_polyps = []
        self.transformations_list = []
        self.transform = None
        
        if transformations:
            self.transformations_list = [
                    'resize',
                    # 'randomHorizontalFlip', 
                    # 'randomVerticalFlip', 
                    # 'randomRotation (30)',
                    'normalize']

            self.transform = transforms.Compose([
                transforms.Resize((224,224), antialias=True),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if one_vs_rest:
            self.dic_label2idx = {'AD': 0, 'ASS': 1, 'HP': 1}
            self.dic_idx2label = {0: 'AD', 1: 'ASS', 1: 'HP'}
        else:
            self.dic_label2idx = {'AD': 0, 'ASS': 1, 'HP': 2}
            self.dic_idx2label = {0: 'AD', 1: 'ASS', 2: 'HP'}
        
        self.df = pd.read_csv(csv_file)
        
        for i, img_id in enumerate(self.df['image_id']):
            img_path = f"{os.path.join(image_dir, str(img_id))}.tif"
            img = Image.open(img_path).convert("RGB")
            self.images.append(img)
            
            if mask_dir:
                mask_path = f"{os.path.join(mask_dir, str(img_id))}.tif"
                mask = Image.open(mask_path).convert("L")
            
                mask = np.array(mask)
                mask = mask > 0
            
                masked_image = np.array(img) * np.expand_dims(mask, axis=-1)
                masked_image = Image.fromarray(masked_image)
                
                self.masks.append(mask)
                self.cropped_polyps.append(masked_image)
            self.labels.append(self.dic_label2idx[self.df['cls'][i]])
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(self.df)} images")
        
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if len(self.cropped_polyps) != 0:
            image = self.cropped_polyps[idx]
        else:
            image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
    
    def visualize_image(self, image, mask=None, masked_image=None):
        if mask is not None and masked_image is not None:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                
            ax[0].imshow(image)
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            
            # Masked region (polyp region only)
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("Mask (Polyp Region)")
            ax[1].axis('off')

            # Image with mask applied
            ax[2].imshow(masked_image)
            ax[2].set_title("Masked Image (Polyp Extracted)")
            ax[2].axis('off')
        
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            
            ax.imshow(image)
            ax.set_title("Original Image")
            ax.axis('off')
        
        plt.show()