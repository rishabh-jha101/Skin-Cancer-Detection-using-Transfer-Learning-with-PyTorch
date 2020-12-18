import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class SkinCancerDataset(Dataset):
    """Skin Cancer Benign Vs Malignant Dataset"""
    def __init__(self, data_dict, data_dir, transforms = None):
        """Initialization

        Args:
        data_dict: dictionary of all the data in the form
                    {"sample": "label"}
        data_dir: directory of the stored dataset
        transform: transforms to be applied on the inputs (default:None)

        """
        self.info = pd.DataFrame(data_dict)
        self.root = data_dir
        self.transforms = transforms
    def __len__(self):
        """Returns:

            Length of the dataset
        """
        return len(self.info)

    def __getitem__(self, idx):
        """Args:

            idx: Index of the asked item

           Returns:

           A tuple of an instance and its labels
            """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root,
                                self.info.iloc[idx, 0])+".jpg"
        image = Image.open(img_name).convert("RGB")
        if self.transforms is not None:
            img_as_tensor = self.transforms(image)
        label = self.info.iloc[idx, 1]

        return (img_as_tensor, torch.tensor(int(label)))
