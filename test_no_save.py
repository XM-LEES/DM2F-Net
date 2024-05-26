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

pth = {
    # OHaze
    # 'snapshot0': 'iter_2000_loss_0.04959_lr_0.000164',
    # 'snapshot1': 'iter_4000_loss_0.04835_lr_0.000126',
    # 'snapshot2': 'iter_6000_loss_0.04916_lr_0.000088',
    # 'snapshot3': 'iter_8000_loss_0.04845_lr_0.000047',


    # 'snapshot7': 'iter_22000_loss_0.01520_lr_0.000244',
    # 'snapshot8': 'iter_24000_loss_0.01475_lr_0.000219',
    # 'snapshot9': 'iter_26000_loss_0.01494_lr_0.000194',
    # 'snapshot10': 'iter_28000_loss_0.01381_lr_0.000169',
    # 'snapshot11': 'iter_30000_loss_0.01353_lr_0.000144',
    # 'snapshot12': 'iter_32000_loss_0.01457_lr_0.000117',
    # 'snapshot13': 'iter_34000_loss_0.01241_lr_0.000091',
    # 'snapshot14': 'iter_36000_loss_0.01224_lr_0.000063',
    # 'snapshot15': 'iter_38000_loss_0.01237_lr_0.000034',

    # RESIDE_ITS_LOSS
    # 'snapshot0': 'iter_8000_loss_0.02054_lr_0.000409',
    # 'snapshot1': 'iter_10000_loss_0.01821_lr_0.000386',
    # 'snapshot2': 'iter_12000_loss_0.02003_lr_0.000363',
    # 'snapshot3': 'iter_14000_loss_0.01685_lr_0.000339',
    # 'snapshot4': 'iter_16000_loss_0.01672_lr_0.000316',
    # 'snapshot5': 'iter_18000_loss_0.01579_lr_0.000292',
    # 'snapshot6': 'iter_20000_loss_0.01750_lr_0.000268',
    # 'snapshot7': 'iter_22000_loss_0.01520_lr_0.000244',
    # 'snapshot8': 'iter_24000_loss_0.01475_lr_0.000219',
    # 'snapshot9': 'iter_26000_loss_0.01494_lr_0.000194',
    # 'snapshot10': 'iter_28000_loss_0.01381_lr_0.000169',
    # 'snapshot11': 'iter_30000_loss_0.01353_lr_0.000144',
    # 'snapshot12': 'iter_32000_loss_0.01457_lr_0.000117',
    # 'snapshot13': 'iter_34000_loss_0.01241_lr_0.000091',
    # 'snapshot14': 'iter_36000_loss_0.01224_lr_0.000063',
    # 'snapshot15': 'iter_38000_loss_0.01237_lr_0.000034',
    # 'snapshot16': 'iter_40000_loss_0.01306_lr_0.000000',
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

            args = {}
            for i in range(len(pth)):
                print (i)
                args['snapshot'] = pth[f'snapshot{i}']
                print(args)
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
