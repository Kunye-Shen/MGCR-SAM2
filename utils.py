import os
import numpy as np
import pytorch_iou
import pytorch_ssim
from PIL import Image
import torch.nn as nn


bce_loss = nn.BCELoss(size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)

def hybrid_loss(pred,target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss

def save_output(img_dir, pred, save_dir):
    img_name = img_dir.split('/')[-1][:-4]
    img = np.array(Image.open(img_dir).convert('RGB'))
    h, w, _ = img.shape

    pred = (pred*255).cpu().detach().numpy()
    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize((w, h), resample=Image.BILINEAR)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    pred.save(f'{save_dir}{img_name}.png')
    
    return