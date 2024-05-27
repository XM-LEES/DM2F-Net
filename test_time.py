# coding: utf-8
import os
import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazeRDDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage import color
from skimage.color import deltaE_ciede2000


def calculate_ciede2000(image1, image2):
    lab1 = color.rgb2lab(image1)
    lab2 = color.rgb2lab(image2)
    delta_e = deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS_LOSS'
exp_name = 'O-Haze'

args = {
    # RESIDE_ITS_LOSS
    # 'snapshot': 'iter_22000_loss_0.01520_lr_0.000244',

    # O-Haze
    'snapshot': 'iter_20000_loss_0.04916_lr_0.000088',
}

to_test = {
    # 'HazeRD': HAZERD_ROOT,
    'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, '')
            elif 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazeRDDataset(root)
            else:
                raise NotImplementedError

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            loss_record = AvgMeter()

            total_time = 0  # 初始化总时间

            for i, data in enumerate(dataloader):
                start_time = time.time()  # 开始计时

                haze, gts, fs = data
                check_mkdir(os.path.join(ckpt_path, exp_name,
                                        '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
                haze = haze.cuda()

                res = net(haze).detach()
                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                end_time = time.time()  # 结束计时
                run_time = end_time - start_time  # 计算本次循环运行时间
                total_time += run_time  # 累积总时间

                print(f'Processing time for image {i+1}: {run_time:.3f} seconds.')

            print(f'Total processing time for {len(dataloader)} images: {total_time:.3f} seconds.')

main()


if __name__ == '__main__':
    main()
