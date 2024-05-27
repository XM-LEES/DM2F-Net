# coding: utf-8
import os

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
exp_name = 'RESIDE_ITS_LOSS'
exp_name = 'O-Haze'

args = {
    # RESIDE_ITS_LOSS
    'snapshot': 'iter_22000_loss_0.01520_lr_0.000244',

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
                # net = DM2FNet_woPhy().cuda()
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

            mses, psnrs, ssims, ciede2000s = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                        '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    
                    # Additions
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                    ssims.append(ssim)
                    ciede2000 = calculate_ciede2000(r, gt)
                    ciede2000s.append(ciede2000)
                    
                    print('predicting for {} ({}/{}) [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                        .format(name, idx + 1, len(dataloader), fs[i], mse, psnr, ssim, ciede2000))
                    log = f'predicting for {name} ({idx + 1}/{len(dataloader)}) [{fs[i]}]: MSE {mse:.4f}, PSNR {psnr:.4f}, SSIM {ssim:.4f}, CIEDE2000 {ciede2000:.4f}'
                    
                    
                    log_path = os.path.join(ckpt_path, exp_name,
                                            '(%s) %s_%s' % (exp_name, name, args['snapshot']), f'{args["snapshot"]}_{name}_result.txt')
                    open(log_path, 'a').write(log + '\n')

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                    '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}")
            
            log = f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}"
            log_path = os.path.join(ckpt_path, exp_name,
                                            f'{name}_avg_result.txt')
            open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()
