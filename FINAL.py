import os
import csv
import scipy.io
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from pytorch_wavelets import DWTForward
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim1, mean_squared_error
import ctlib
import odl
from odl.contrib.torch import OperatorModule
from networks import NFFB
from utils import get_data_loader
from basicsr.models.archs.LEARN import LEARN
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import matplotlib.pyplot as plt
# Set parameters
views = 12
dets = 624
width = 256
height = 256
dImg = 0.006641
dDet = 0.012858
Ang0 = 0.0
dAng = 2 * np.pi / views
s2r = 5.95
d2r = 4.906
binshift = -0.0013
scan_type = 0

options = torch.tensor([views, dets, width, height, dImg, dDet, Ang0, 2 * np.pi / views, s2r, d2r, binshift, scan_type]).cuda()

def tv1_loss(x):
    ndims = len(list(x.size()))
    if ndims != 4:
        assert False, "Input must be {batch, channel, height, width}"
    n_pixels = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    tot_var = torch.sum(dh) + torch.sum(dw)
    tot_var = tot_var / n_pixels
    return tot_var

class ProjOperator:
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=95,
                 det_origin=20):
        self.N = N
        self.M = M
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.det_pixels = det_pixels
        self.det_pixel_size = det_pixel_size
        self.angles = angles
        self.src_origin = src_origin
        self.det_origin = det_origin

    def forward(self, img):
        reco_space = odl.uniform_discr(
            min_pt=[-20, -20], max_pt=[20, 20], shape=[self.N, self.M],
            dtype='float32')
        angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
        detector_partition = odl.uniform_partition(-60, 60, self.det_pixels)
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

        if isinstance(img, torch.Tensor):
            fwd_op_mod = OperatorModule(ray_trafo)
            proj_data = fwd_op_mod(img)
        else:
            proj_data = ray_trafo(img)

        return proj_data

    def __call__(self, img):
        return self.forward(img)

class FBPOperator:
    def __init__(self, N=512, M=512, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=720, src_origin=950,
                 det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7):
        self.N = N
        self.M = M
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.det_pixels = det_pixels
        self.det_pixel_size = det_pixel_size
        self.angles = angles
        self.src_origin = src_origin
        self.det_origin = det_origin
        self.filter_type = filter_type
        self.frequency_scaling = frequency_scaling

        self.reco_space = odl.uniform_discr(
            min_pt=[-20, -20], max_pt=[20, 20], shape=[self.N, self.M],
            dtype='float32')
        self.angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
        self.detector_partition = odl.uniform_partition(-60, 60, self.det_pixels)
        self.geometry = odl.tomo.FanBeamGeometry(
            self.angle_partition, self.detector_partition,
            src_radius=self.src_origin,
            det_radius=self.det_origin)
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, self.geometry)
        self.fbp = odl.tomo.fbp_op(self.ray_trafo, filter_type=self.filter_type,
                                   frequency_scaling=self.frequency_scaling)

    def forward(self, proj):
        if isinstance(proj, torch.Tensor):
            parker_weighted_fbp_mod = OperatorModule(self.fbp)
            reconstructed_img = parker_weighted_fbp_mod(proj)
        else:
            reconstructed_img = self.fbp(proj).data

        return reconstructed_img

    def __call__(self, proj):
        return self.forward(proj)

radon_pro = ProjOperator(N=256, M=256, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=20, src_origin=59,
                 det_origin=49)

radon_pro_60 = ProjOperator(N=256, M=256, pixel_size_x=0.15, pixel_size_y=0.15,
det_pixels=624, det_pixel_size=0.2, angles=60, src_origin=59,
det_origin=49)

fbp_op = FBPOperator(N=256, M=256, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=20, src_origin=59,
                 det_origin=49, filter_type='Hann', frequency_scaling=1)

def compare(recon0, recon1, verbose=True):
    mse_recon = mean_squared_error(recon0, recon1)
    small_side = np.min(recon0.shape)
    if small_side < 7:
        if small_side % 2:
            win_size = small_side
        else:
            win_size = small_side - 1
    else:
        win_size = None

    ssim_recon = ssim1(recon0, recon1,
                       data_range=recon0.max() - recon0.min(), win_size=win_size)
    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                                         data_range=recon0.max() - recon0.min())

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)

def clear(x):
    return x.detach().cpu().squeeze().numpy()

# Define the configuration dictionary
config = {
    "encoding": {
        "feat_dim": 2, 
        "base_resolution": 64,
        "per_level_scale": 2,
        "base_sigma": 5.0,
        "exp_sigma": 2.0,
        "grid_embedding_std": 0.01
    },
    "SIREN": {
        "dims": [128, 128, 128, 128],
        "w0": 100,
        "w1": 100,
        "size_factor": 1
    },
    "Backbone": {
        "dims": [64, 64, 64, 64]
    },
    "log_iter": 10,
    "val_iter": 100,
    "max_iter": 2000,
    "batch_size": 1,
    "loss": "L2",
    "optimizer": "Adam",
    "weight_decay": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
    "lr": 0.00001,
    "data": "pancs_4dct_phase6",
    "img_size": 256,
    "img_slice": 24,
    "img_path": "./data/ct_data/pancs_4dct_phase6.npz"
}

# Initialize the model
model = NFFB(config, out_dims=1)
model.cuda()
model.train()

# Load pretrained model
Mamba =  LEARN(block_num=10).cuda()
Mamba.load_state_dict(torch.load('/home/wujia/daima/NeRP-main/上传代码/pre_trainmodel/best_model.pth'))
# Mamba = LEARN_FBP(block_num=50, options=options).cuda()
# Mamba.load_state_dict(torch.load('/home/wujia/daima/NeRP-main/duibi/train_model/best_learn_12.pth'))
Mamba.eval()

# Setup optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.99), weight_decay=0.00001)
scheduler = CosineAnnealingLR(optim, T_max=3000, eta_min=1e-6)

# Setup loss functions
loss_fn1 = torch.nn.MSELoss()
loss_fn2 = torch.nn.L1Loss()
dwt = DWTForward(J=3, wave='db1', mode='zero').cuda()

# Setup data loader
data_loader = get_data_loader(config['data'], '/home/wujia/daima/NeRP-main/duibi/NeRP/data/ct_data/L067_FD_3_1.CT.0002.0151.2015.12.22.18.12.07.5968.358095025.IMA', config['img_size'], config['img_slice'], train=True, batch_size=config['batch_size'])

# # Setup for saving results
# results_csv_file = '/home/wujia/daima/NeRP-main/第二个实验验证多尺度神经常/123.csv'
# if not os.path.exists(results_csv_file):
#     with open(results_csv_file, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Iteration', 'MSE', 'SSIM', 'PSNR'])
best_psnr = 0

for it, (grid, image) in enumerate(data_loader):
    grid = grid.cuda()
    image = image.cuda()

    projs = radon_pro(image.permute(0, 3, 1, 2))
    
    ima_prior = fbp_op(projs)
    # projs_ctlib = ctlib.projection(image.permute(0, 3, 1, 2), options)
    # ima_prior = ctlib.fbp(projs_ctlib, options)

    with torch.no_grad():
        img_prior = Mamba(ima_prior, projs)
        plt.imshow(clear(img_prior), cmap='gray')
        plt.show()
        img_prior = img_prior.permute(0, 2, 3, 1)
        compare(clear(img_prior), clear(image))
        # scipy.io.savemat('/home/wujia/daima/NeRP-main/duibi/12对比结果/5025/img_prior.mat', {'recon': clear(img_prior)})


    test_data = (grid, image)
    train_data = (grid, projs)

    for iterations in range(3000):
        model.train()
        optim.zero_grad()

        train_embedding = train_data[0].reshape(-1, 2)
        train_output1 = model(train_embedding)
        train_output1 = train_output1.reshape(1, 256, 256, 1)

        train_output = train_output1 + test_data[1]
        train_projs = radon_pro(train_output.permute(0, 3, 1, 2))
        Yl, Yh = dwt(train_output1.permute(0, 3, 1, 2))
        wavelet_l1_norm = torch.norm(Yh[0], dim=2).mean()

        train_loss = loss_fn1(train_projs, train_data[1]) + 0.1 * (0.1 * wavelet_l1_norm + 0.9 * tv1_loss(train_output1.permute(0, 3, 1, 2)))

        train_loss.backward()
        optim.step()
        # scheduler.step()

        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()

            mse, ssim, psnr = compare(clear(test_data[1]), clear(train_output))

            # with open(results_csv_file, 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([iterations + 1, mse, ssim, psnr])

            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, 2000, train_loss, psnr))

            if psnr > best_psnr:
                best_psnr = psnr
                recon_train_output = clear(train_output)
                # scipy.io.savemat(f'/home/wujia/daima/NeRP-main/duibi/12对比结果/5025/1219_new.mat', {'recon': recon_train_output})
                plt.imsave(f'/home/wujia/daima/NeRP-main/img/recon_best_psnr.png', recon_train_output, cmap='gray')

                recon_test_data = clear(test_data[1])
