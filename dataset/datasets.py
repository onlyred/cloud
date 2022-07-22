import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from  typing import List

class CustomDataset(Dataset):
    def __init__(self, data_path : str, resize : int = 227):
        super(CustomDataset, self).__init__()
        
        self.data_path = data_path
        
        x, y = create_dataset(self.data_path, resize)

        labels    = os.listdir(data_path)
        dataset_label_encoder = make_labels(labels)

        self.x = x
        self.y = np.array(dataset_label_encoder.transform(y))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx]/255.).float()
        y = torch.tensor(self.y[idx])
        return x, y

def create_dataset(data_path : str, resize : int = 227):
    x, y = [], []
    if not os.path.exists(data_path):
        print(f"Can't find : {data_path}")
        sys.exit(1)

    labels    = os.listdir(data_path)
    transform = transforms.Resize(size = resize)
    for label in labels:
        file_lists = os.listdir(os.path.join(data_path, label))
        for fname in file_lists:
            if os.path.exists(fname):
                print(f"Can't find : {fname}")
                sys.exit(1)

            img = cv2.imread(os.path.join(data_path, label, fname))
            if img.shape[0] == resize and img.shape[1] == resize:
                img = np.transpose(img, [2,0,1])
                continue
            else:
                # Convert from cv2_img to pil_img for transfroming
                color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                pil_img = Image.fromarray(color_converted)  
                img = transform(pil_img)
                img = np.array(img)
                # Show image
                #_, ax = plt.subplots(nrows=1,ncols=2, figsize=(6,3))
                #ax[0].imshow(pil_img)
                #ax[1].imshow(img)
                #plt.show()
                img = np.transpose(img, [2,0,1])
            x.append(img)
            y.append(label)
    return np.array(x), np.array(y)

def make_labels(labels: List):
    dataset_label_encoder = LabelEncoder()
    dataset_label_encoder.fit(labels)
    return dataset_label_encoder

#if __name__ == "__main__":
#    data_path = "/mnt/ai-nas02/TEAM_PROJECT/Cloud/TypeClassification/CCSN_v2"
#    cds = create_dataset(data_path)
