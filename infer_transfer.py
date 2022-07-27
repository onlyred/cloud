import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_argments():
    parser = argparse.ArgumentParser(description='cloud-type-classification')
    parser.add_argument('--pth', type=str, default='./best.pth')
    #parser.add_argument('--resize', type=int, default=227)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--classes', type=str, default='class.txt')

    return parser.parse_args()

def main():
    args = get_argments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize(227),
                    T.ToTensor(),
                ])

    try:
        img = cv2.imread(args.filename)
        img = transform(img)
        img = img.unsqueeze(axis=0)
    except Exception as e:
        print(f"Error : {e}")

    # load model
    net = torchvision.models.alexnet(pretrained=True)
    net.classifier[6] = nn.Linear(4096, args.num_classes)

    net.load_state_dict(torch.load(args.pth))
    net.to(device)

    predict = net(img.to(device))
    print(predict)
    idx = predict.argmax(axis=1).detach().cpu().numpy()[0]
    # read class
    with open(args.classes, 'r') as f:
        cls = f.readlines()

    print(f"predicted class : {idx} = {cls[idx]}")
    # plot image
    img = img.squeeze().permute(1,2,0)
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

if __name__ == "__main__":
    main()


