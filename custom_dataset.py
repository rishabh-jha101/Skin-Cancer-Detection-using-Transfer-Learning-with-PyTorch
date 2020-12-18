import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class SkinCancerDataset(Dataset):
    def __init__(self, data_dict, data_dir, transforms = None):

        self.info = pd.DataFrame(data_dict)
        self.root = data_dir
        self.transforms = transforms
    def __len__(self):

        return len(self.info)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root,
                                self.info.iloc[idx, 0])+".jpg"
        image = Image.open(img_name).convert("RGB")
        if self.transforms is not None:
            img_as_tensor = self.transforms(image)
        label = self.info.iloc[idx, 1]        

        return (img_as_tensor, torch.tensor(int(label)))
