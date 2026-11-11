import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from utils import *
from data_loader import *
from model import MGCRSAM2


def main(args):
    # ------- model setting -------
    net = MGCRSAM2()
    net.load_state_dict(torch.load(args.model_dir))

    if args.parallel == True:
        net = nn.DataParallel(net, device_ids=args.cuda_device)
    
    if torch.cuda.is_available():
        net.cuda()

    test(net, args)

    return

def test(net, args):
    # ------- load testing data -------
    transform_test = transforms.Compose([
                Resize((args.img_size, args.img_size)),
                ToTensor(),
                Normalize()
            ])
    
    # ------- start testing -------
    print("------- Start Testing -------")

    net.eval()
    # load testing data
    print(f"Dataset: {args.dataset}")
    dataset_test = DefectDataset(args.dataset_path, transform_test, 'test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for index, data in tqdm(enumerate(dataloader_test), total=dataloader_test.__len__()):
        image, name = data['image'], data['name']

        image = image.type(torch.FloatTensor)

        if torch.cuda.is_available():
            image = Variable(image.cuda(), requires_grad=False)
        else:
            image = Variable(image, requires_grad=False)

        pr1, pr2, pr3, p1, p2, p3, p4 = net(image)

        for i in range(pr1.shape[0]):
            pr1_i = pr1[i,0,:,:]
            pr1_i = normPRED(pr1_i)
            save_output(name[i], pr1_i, args.pre_dir)

        del image, pr1, pr2, pr3, p1, p2, p3, p4, pr1_i

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--CUDA", type=str, default='0,1')
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--parallel", type=bool, default=True)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--cuda_device", type=list, default=[0,1])

    # root
    p.add_argument("--dataset", type=str, default='SSP2000')
    p.add_argument("--dataset_path", type=str, default="/Dataset/")
    p.add_argument("--model_dir", type=str, default="./model_save/")
    p.add_argument("--pre_dir", type=str, default="./predicts/")
    args = p.parse_args()

    args.pre_dir = os.path.join(args.pre_dir, args.dataset, '')
    args.dataset_path = os.path.join(args.dataset_path, args.dataset, '')
    args.model_dir = os.path.join(args.model_dir, args.dataset, 'MGCRSAM2.pth')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA

    main(args)