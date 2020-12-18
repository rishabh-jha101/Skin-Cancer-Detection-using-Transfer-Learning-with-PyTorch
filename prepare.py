# from torchvision import datasets
import numpy as np
import json
from custom_dataset import SkinCancerDataset
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

test_size = 0.1

# convert data to a normalized torch.FloatTensor and Resize
transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # Imagenet standards
    ])
#data directory
data_dir = "C:/Users/RJ/Desktop/Deep Learning/Projects/skin_cancer_transfer/data/"
with open('labels.json') as f:
    targets = json.load(f)

# choose the training and test datasets with transforms
train_data = SkinCancerDataset(targets.items(), data_dir, transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

#Slicing Train/Valid/Test in 70/20/10 ratio
split1 = int(np.floor(test_size * num_train))
split2 = int(np.floor(valid_size * num_train))
test_idx, valid_idx, train_idx = indices[:split1], indices[split1:split2], indices[split2:]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=test_sampler, num_workers=num_workers)
