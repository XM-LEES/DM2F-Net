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

args = {
    # RESIDE_ITS_LOSS
    'snapshot': 'iter_22000_loss_0.01520_lr_0.000244',

    # O-Haze
    'snapshot': 'iter_20000_loss_0.04916_lr_0.000088',
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
