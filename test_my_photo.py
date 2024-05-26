# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, MYHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, MyHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'RESIDE_ITS_LOSS'
# exp_name = 'O-Haze'

pth = {
    'snapshot1': 'iter_14000_loss_0.01685_lr_0.000339',
    'snapshot2': 'iter_16000_loss_0.01672_lr_0.000316',
    'snapshot3': 'iter_18000_loss_0.01579_lr_0.000292',
    'snapshot4': 'iter_20000_loss_0.01750_lr_0.000268',
    'snapshot5': 'iter_22000_loss_0.01520_lr_0.000244',
    'snapshot6': 'iter_24000_loss_0.01475_lr_0.000219',
    'snapshot7': 'iter_26000_loss_0.01494_lr_0.000194',
    'snapshot8': 'iter_28000_loss_0.01381_lr_0.000169',
    'snapshot9': 'iter_30000_loss_0.01353_lr_0.000144',
}

to_test = {
    'MyHaze': MYHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        for name, root in to_test.items():
            if 'MyHaze' in name:
                net = DM2FNet().cuda()
                dataset = MyHazeDataset(root)
            else:
                raise NotImplementedError

            args = {}
            for i in range(len(pth)):
                print (i)
                args['snapshot'] = pth[f'snapshot{i + 1}']
                
                if len(args['snapshot']) > 0:
                    print('load snapshot \'%s\' for testing' % args['snapshot'])
                    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

                net.eval()
                dataloader = DataLoader(dataset, batch_size=1)

                for idx, data in enumerate(dataloader):
                    haze, fs = data

                    check_mkdir(os.path.join(ckpt_path, exp_name,
                                            '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
                    haze = haze.cuda()

                    if 'O-Haze' in name:
                        res = sliding_forward(net, haze).detach()
                    else:
                        res = net(haze).detach()

                    for r, f in zip(res.cpu(), fs):
                        to_pil(r).save(
                            os.path.join(ckpt_path, exp_name,
                                        '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

if __name__ == '__main__':
    main()
