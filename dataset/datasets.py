import torch
import torchvision.transforms as T
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
        self.resize    = resize
        self.transform = T.Compose([
                             T.ToPILImage(),
                             T.RandomResizedCrop(resize),
                             T.ColorJitter(brightness=0.3, saturation=0.3),
                             T.RandomHorizontalFlip(),
                             T.ToTensor(),
                         ])
        
        if not os.path.exists(data_path):
            print(f"Can't find : {data_path}")
            sys.exit(1)
        
        self.labels = os.listdir(data_path)

        # save classes
        with open('class.txt','w') as f:
            for l in self.labels:
                f.write(f"{l}\n")

        self.x = []
        self.y = []
        for label in self.labels:
             file_lists =  os.listdir(os.path.join(data_path, label))
             for fname in file_lists:
                 ## original
                 self.x.append(self.preprocessing(os.path.join(data_path, label, fname),
                                                  resize))
                 self.y.append(label)
                 ## data augmentation
                 for i in range(9):
                     self.x.append(self.preprocessing(os.path.join(data_path, label, fname),
                                                      resize, self.transform))
                     self.y.append(label)
        # encoding labels
        dataset_label_encoder = self.make_labels(self.labels)
        self.y = dataset_label_encoder.transform(self.y)

    @staticmethod
    def make_labels(labels: List):
        dataset_label_encoder = LabelEncoder()
        dataset_label_encoder.fit(labels)
        return dataset_label_encoder

    @staticmethod
    def preprocessing(fname : str, resize : int, transform : T = None):
        if not os.path.exists(fname):
            print(f"Can't find : {fname}")
            sys.exit(1)

        img = cv2.imread(fname)
        if transform is None:
            transform = T.Compose([
                            T.ToPILImage(),
                            T.Resize(resize),
                            T.ToTensor(),  # conver RGB pixel value in the [0,255] to [0.0, 1.0] range
                        ])
        img = transform(img)

        ## plot image
        #plt.imshow(img.permute(1,2,0))
        #plt.show(block=False)
        #plt.pause(1)
        return img

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], torch.tensor(self.y[idx])
