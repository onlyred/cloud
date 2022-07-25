import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import CustomDataset
from model import AlexNet
from util import AverageMeter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam


def get_argments():
    parser = argparse.ArgumentParser(description='cloud-type-classification')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=227)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--file_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_argments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataloader
    dataset = CustomDataset(args.file_path, args.resize)

    num_data = len(dataset)
    num_train = int(num_data * 0.7)
    num_valid = num_data - num_train
    train_set, valid_set = random_split(dataset,[num_train,num_valid])

    train_loader = DataLoader(train_set,
                              batch_size = args.batchsize,
                              shuffle=True,
                              num_workers = args.num_workers)

    valid_loader = DataLoader(valid_set,
                              batch_size = args.batchsize,
                              shuffle=True,
                              num_workers = args.num_workers)
    # load model
    net = AlexNet(in_channel=3, 
                  out_channel=args.num_classes, 
                  dropout=0.3)

    net.to(device)
    optim = SGD(net.parameters(), 
                lr=args.learning_rate,
                momentum=0.95,
                weight_decay=0.00005)
    #optim = Adam(net.parameters(), 
    #             lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(args, net, optim, criterion, 
          train_loader, valid_loader, args.patience, device)

def train(args, model, optimizer, criterion, train_loader, valid_loader, patience, device):
    early_stop = 0
    train_loss = []
    valid_loss = []
    for epoch in range(1,args.epochs+1):
        train_losses = AverageMeter()
        val_losses = AverageMeter()
        valid_corr = 0
        total_corr = 0

        model.train()
        with tqdm(total=len(train_loader) - len(train_loader) % args.batchsize) as t:
            t.set_description(f"epoch : {epoch}/{args.epochs}")

            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)

                loss  = criterion(preds, labels)

                train_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=f'{train_losses.avg:.4f}')
                t.update(len(inputs))

        train_loss.append(train_losses.avg)

        model.eval()
        for data in valid_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                output = model(inputs)
                loss  = criterion(output, labels)
                _, preds = torch.max(output.data, 1) # return values, indices 
                valid_corr += (preds == labels).sum().item()
                total_corr += labels.size(0)
                correct = 100. * valid_corr / total_corr

            val_losses.update(loss.item(), len(inputs))

        valid_loss.append(val_losses.avg)

        if epoch == 1:
            best_loss = val_losses.avg
            best_corr = correct
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
        elif correct > best_corr:  # max
            best_epoch = epoch
            best_loss = val_losses.avg
            best_corr = correct
            best_model = copy.deepcopy(model.state_dict())
            early_stop = 0
        else:                             # earlystopping
            early_stop += 1
            if patience < early_stop:
                print(f'early stopping : stop training')
                break

        print(f'eval : {val_losses.avg:.6f}, best correct : {best_corr:.2f}% ( {early_stop} / {patience} )')

    print(f'best epoch: {best_epoch}, loss: {best_loss:.9f}')
    
    plot_learning_curve(train_loss, valid_loss, epoch)
    torch.save(best_model, 'best.pth')

def plot_learning_curve(train_loss, valid_loss, epoch):
    plt.figure(figsize=(5,5))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(np.arange(1,epoch+1), train_loss, '-', color='b', label='train')
    plt.plot(np.arange(1,epoch+1), valid_loss, '-', color='r', label='valid')
    plt.legend(loc='best')
    plt.savefig('learning_curve.png')
    plt.close()

if __name__ == "__main__":
    main()


