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




os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'RESIDE_ITS_LDP'
# exp_name = 'O-Haze'

args = {
    # 'snapshot': 'iter_2000_loss_0.03465_lr_0.000477',
    # 'snapshot': 'iter_4000_loss_0.02477_lr_0.000455',
    # 'snapshot': 'iter_6000_loss_0.02056_lr_0.000432',
    # 'snapshot': 'iter_8000_loss_0.02018_lr_0.000409',
    'snapshot': 'iter_10000_loss_0.01882_lr_0.000386',
    # 'snapshot': 'iter_12000_loss_0.02151_lr_0.000363',
    # 'snapshot': 'iter_14000_loss_0.01666_lr_0.000339',
    # 'snapshot': 'iter_16000_loss_0.01553_lr_0.000316',
    # 'snapshot': 'iter_18000_loss_0.01595_lr_0.000292',
    # 'snapshot': 'iter_20000_loss_0.01554_lr_0.000268',
    # 'snapshot': 'iter_22000_loss_0.01464_lr_0.000244',
    # 'snapshot': 'iter_24000_loss_0.01288_lr_0.000219',
    # 'snapshot': 'iter_26000_loss_0.01384_lr_0.000194',
    # 'snapshot': 'iter_28000_loss_0.01337_lr_0.000169',
    # 'snapshot': 'iter_30000_loss_0.01315_lr_0.000144',
    # 'snapshot': 'iter_32000_loss_0.01461_lr_0.000117',
    # 'snapshot': 'iter_34000_loss_0.01279_lr_0.000091',
    # 'snapshot': 'iter_36000_loss_0.01237_lr_0.000063',
    # 'snapshot': 'iter_38000_loss_0.01227_lr_0.000034',
    # 'snapshot': 'iter_40000_loss_0.01195_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'HazeRD': HAZERD_ROOT,
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
                net = DM2FNet().cuda()
                dataset = OHazeDataset(root, '')
            elif 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazeRDDataset(root)
            else:
                raise NotImplementedError

        #     # net = nn.DataParallel(net)



            for arg in ['iter_2000_loss_0.03465_lr_0.000477', 'iter_4000_loss_0.02477_lr_0.000455', 'iter_6000_loss_0.02056_lr_0.000432', 'iter_8000_loss_0.02018_lr_0.000409', 
                               'iter_10000_loss_0.01882_lr_0.000386', 'iter_12000_loss_0.02151_lr_0.000363', 'iter_14000_loss_0.01666_lr_0.000339', 'iter_16000_loss_0.01553_lr_0.000316',
                               'iter_18000_loss_0.01595_lr_0.000292', 'iter_20000_loss_0.01554_lr_0.000268', 'iter_22000_loss_0.01464_lr_0.000244', 'iter_24000_loss_0.01288_lr_0.000219', 
                               'iter_26000_loss_0.01384_lr_0.000194', 'iter_28000_loss_0.01337_lr_0.000169', 'iter_30000_loss_0.01315_lr_0.000144', 'iter_32000_loss_0.01461_lr_0.000117',
                                'iter_34000_loss_0.01279_lr_0.000091', 'iter_36000_loss_0.01237_lr_0.000063', 'iter_38000_loss_0.01227_lr_0.000034']:
                args['snapshot'] = arg


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
                        # print('predicting for {} ({}/{}) [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}'
                        #       .format(name, idx + 1, len(dataloader), fs[i], mse, psnr, ssim))
                        log = f'predicting for {name} ({idx + 1}/{len(dataloader)}) [{fs[i]}]: MSE {mse:.4f}, PSNR {psnr:.4f}, SSIM {ssim:.4f}'
                        
                        
                        log_path = os.path.join(ckpt_path, exp_name,
                                                '(%s) %s_%s' % (exp_name, name, args['snapshot']), f'{args["snapshot"]}_{name}_result.txt')
                        open(log_path, 'a').write(log + '\n')

                    # for r, f in zip(res.cpu(), fs):
                    #     to_pil(r).save(
                    #         os.path.join(ckpt_path, exp_name,
                    #                      '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

                print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}")
                
                log = f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}"
                log_path = os.path.join(ckpt_path, exp_name,
                                                '(%s) %s_%s' % (exp_name, name, args['snapshot']), f'{args["snapshot"]}_{name}_avg_result.txt')
                open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()
