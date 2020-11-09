import argparse
from time import gmtime, strftime
import os
import wandb
import torch
from torch.utils.data import DataLoader
import torchvision.models as models

from arch.preresnext import *
from batch_manager import BatchManagerTinyImageNet
import train
import val
from simple_tool_pytorch import ImageNetPolicy
import torchvision.transforms as transforms
import torch.nn as nn
from simple_tool_pytorch import GradualWarmupScheduler
from simple_tool_pytorch import RandomErasing

if __name__ == '__main__':
    wandb.init(project="gpt-3")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_base', type=float, default=25e-05)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    wandb.config.update(args)

    # define model
    
#     se_resnext50
    model = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
    net = torch.load('./data/se_resnext50_32x4d-a260b3a4.pth')
    print(' pretrain_model loading...')
    model.load_state_dict(net)
    model.layer0.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.layer0.pool = nn.Sequential()
    model.avg_pool= nn.AdaptiveAvgPool2d((1, 1))
    model.last_linear.out_features = 200
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)
    print(model)
    
    print('==> Preparing data..')
    transform_train = transforms.Compose([
#         transforms.Resize(224),
#         transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         ImageNetPolicy(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
#        RandomErasing(probability=0.5, mean=[0.4802, 0.4481, 0.3975])
        
    ])
    transform_val = transforms.Compose([
#        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4824, 0.4495, 0.3981), (0.2301, 0.2264, 0.2261)),
    ])
    
    # define batch_manager
    dataloader_train = DataLoader(BatchManagerTinyImageNet(split='train',transform=transform_train), 
                                  shuffle=True, num_workers=8, batch_size=args.batch_size)
    dataloader_val = DataLoader(BatchManagerTinyImageNet(split='val',transform=transform_val), 
                                shuffle=False, num_workers=8, batch_size=args.batch_size)

    

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=1e-4)
    
    # LR schedule
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=2, after_scheduler=cosine_scheduler)
    
    wandb.watch(model)
    
    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs+1):
        for param_group in optimizer.param_groups:
            lr_per_epoch = param_group['lr'] 
        print(f"Training at epoch {epoch}. LR {lr_per_epoch}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        scheduler.step(epoch)
        acc1, acc5 = val.val(model, dataloader_val, epoch=epoch)

        save_data = {'epoch': epoch,
                     'acc1': acc1,
                     'acc5': acc5,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        
        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
        if acc1 >= best_perform:
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
            best_perform = acc1
            best_epoch = epoch
        print(f"best performance {best_perform} at epoch {best_epoch}")
