import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import argparse
from tqdm import tqdm

from utils import *
from data_loader import *
from model import MGCRSAM2


def main(args):
    # ------- model setting -------
    # define model
    net = MGCRSAM2(args.hiera_pre)
    
    optimizer = torch.optim.AdamW(net.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    
    # ------- load training data -------
    # data augmentation
    transform_train = transforms.Compose([
                Resize((args.img_size, args.img_size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
    
    dataset_train = DefectDataset(args.dataset_path, transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # ------- start training -------
    print("------- Start Training -------")
    if args.parallel == True:
        net = nn.DataParallel(net, device_ids=args.cuda_device)
        
    if torch.cuda.is_available():
        net.cuda()

    for epoch in range(1, args.epochs+1):
        net.train()
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch: {epoch}/{args.epochs}")
        for index, data in enumerate(dataloader_train):
            image, label = data['image'], data['label']

            image = image.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)

            if torch.cuda.is_available():
                image, label = Variable(image.cuda(), requires_grad=False), Variable(label.cuda(),requires_grad=False)
            else:
                image, label = Variable(image, requires_grad=False), Variable(label,requires_grad=False)

            optimizer.zero_grad()

            pr1, pr2, pr3, p1, p2, p3, p4 = net(image)
            lr1 = hybrid_loss(pr1, label)
            lr2 = hybrid_loss(pr2, label)
            lr3 = hybrid_loss(pr3, label)

            l1 = hybrid_loss(p1, label)
            l2 = hybrid_loss(p2, label)
            l3 = hybrid_loss(p3, label)
            l4 = hybrid_loss(p4, label)
            loss = lr1 + lr2 + lr3 + l1 + l2 + l3 + l4

            loss.backward()
            optimizer.step()

            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss1=lr1.item(), loss=loss.item())

            del image, label, pr1, pr2, pr3, p1, p2, p3, p4, lr1, lr2, lr3, l1, l2, l3, l4, loss

        scheduler.step()

        if args.parallel == True:
            torch.save(net.module.state_dict(), args.save_dir+"epoch_%d.pth"%(epoch))
        else:
            torch.save(net.state_dict(), args.save_dir+"epoch_%d.pth"%(epoch))

    print("------- Training Done -------")

    return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--CUDA", type=str, default='0,1')
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--parallel", type=bool, default=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--cuda_device", type=list, default=[0,1])

    # root
    p.add_argument("--dataset", type=str, default='SSP2000')
    p.add_argument("--dataset_path", type=str, default="/Dataset/")
    p.add_argument("--save_dir", type=str, default="./model_save/")
    p.add_argument("--hiera_pre", type=str, default="./pretrained/sam2_hiera_large.pt")
    args = p.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.dataset, '')
    args.dataset_path = os.path.join(args.dataset_path, args.dataset, '')

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA

    main(args)