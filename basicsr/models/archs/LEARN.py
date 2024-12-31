import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
from .mambairunet_arch import VSSBlock
import numpy as np
from einops import rearrange, repeat
from thop import profile, clever_format
import time
import odl
from odl.contrib.torch import OperatorModule

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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class fidelity_module(nn.Module):
    def __init__(self):
        super(fidelity_module, self).__init__()        
        self.projector = ProjOperator(
            N=256, M=256,
            pixel_size_x=0.15, pixel_size_y=0.15,
            det_pixels=624, det_pixel_size=0.2,
            angles=20,  # 固定角度数
            src_origin=59, det_origin=49
        )
        self.fbp = FBPOperator(
            N=256, M=256,
            pixel_size_x=0.15, pixel_size_y=0.15,
            det_pixels=624, det_pixel_size=0.2,
            angles=20,  # 固定角度数
            src_origin=59, det_origin=49,
            filter_type='Hann', frequency_scaling=1
        )
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        
    def forward(self, input_data, proj):
        p_tmp = self.projector(input_data)
        y_error = proj - p_tmp
        x_error = self.fbp(y_error)
        out = x_error 
        return out

class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module()
        self.block2 = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        self.overlap_patch_embed = OverlapPatchEmbed(in_c=1, embed_dim=hid_channels)
        self.vss_block = VSSBlock(hidden_dim=hid_channels)
        self.channel_reduce = nn.Conv2d(hid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj):
        tmp1 = self.block1(input_data, proj)
        tmp1 = self.block2(tmp1)
        input_embed = self.overlap_patch_embed(input_data)
        tmp3 = self.vss_block(input_embed, input_data.shape[2:4])
        B, _, H, W = input_data.shape
        tmp3 = tmp3.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tmp3 = self.channel_reduce(tmp3)
        output = input_data + tmp1 + tmp3
        output = self.relu(output)
        return output

class LEARN(nn.Module):
    def __init__(self, block_num=10, hid_channels=48, kernel_size=5, padding=2):
        super(LEARN, self).__init__()
        self.model = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding) for i in range(block_num)])
        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_data, proj):
        x = input_data
        for index, module in enumerate(self.model):
            x = module(x, proj)
        return x
    
def validate_input(input_data, proj_data, name=""):
    """验证输入数据的格式和范围并打印信息"""
    print(f"\n{'='*50}")
    print(f"Validating {name} inputs:")
    print(f"{'='*50}")
    
    # 打印基本信息
    print(f"\nInput image:")
    print(f"- Type: {type(input_data)}")
    print(f"- Shape: {input_data.shape}")
    if isinstance(input_data, torch.Tensor):
        print(f"- Device: {input_data.device}")
        print(f"- Value range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    print(f"\nProjection data:")
    print(f"- Type: {type(proj_data)}")
    print(f"- Shape: {proj_data.shape}")
    if isinstance(proj_data, torch.Tensor):
        print(f"- Device: {proj_data.device}")
        print(f"- Value range: [{proj_data.min():.3f}, {proj_data.max():.3f}]")
    
    try:
        # 检查类型
        if not isinstance(input_data, torch.Tensor):
            raise ValueError(f"Input image must be torch.Tensor")
        if not isinstance(proj_data, torch.Tensor):
            raise ValueError(f"Projection data must be torch.Tensor")
        
        # 检查维度
        if input_data.dim() != 4:
            raise ValueError(f"Input image must be 4D tensor [B,C,H,W], got {input_data.dim()}D")
        if proj_data.dim() != 4:
            raise ValueError(f"Projection data must be 4D tensor [B,C,angles,dets], got {proj_data.dim()}D")
        
        # 检查通道数
        if input_data.shape[1] != 1:
            raise ValueError(f"Input image must have 1 channel, got {input_data.shape[1]}")
        if proj_data.shape[1] != 1:
            raise ValueError(f"Projection data must have 1 channel, got {proj_data.shape[1]}")
        
        # 检查尺寸
        if input_data.shape[2:] != (256, 256):
            raise ValueError(f"Input image must be 256x256, got {input_data.shape[2:]}")
        if proj_data.shape[2:] != (20, 624):
            raise ValueError(f"Projection data must be 20x624, got {proj_data.shape[2:]}")
        
        # 检查batch size匹配
        if input_data.shape[0] != proj_data.shape[0]:
            raise ValueError(f"Batch size mismatch: input_data {input_data.shape[0]} vs proj_data {proj_data.shape[0]}")
        
        # 检查数值范围
        if torch.any(input_data < 0):
            raise ValueError(f"Input image contains negative values")
        if torch.any(proj_data < 0):
            raise ValueError(f"Projection data contains negative values")
            
        print(f"\nValidation passed! ✓")
        print(f"{'='*50}\n")
        
    except ValueError as e:
        print(f"\nValidation failed! ✗")
        print(f"Error: {str(e)}")
        print(f"{'='*50}\n")
        raise
    

# if __name__ == '__main__':
#     # 创建模拟数据
#     batch_size = 2  # 测试多个batch
#     input_image = torch.randn(batch_size, 1, 256, 256) * 0.1 + 0.5  # 生成随机值在[0.4,0.6]左右
#     proj_data = torch.randn(batch_size, 1, 20, 624) * 0.1 + 0.5    # 生成随机值在[0.4,0.6]左右
    
#     print("\nCreating test data...")
#     print(f"Input image shape: {input_image.shape}")
#     print(f"Projection data shape: {proj_data.shape}")
    
#     # 创建模型
#     print("\nInitializing LEARN model...")
#     model = LEARN(block_num=10, hid_channels=48, kernel_size=5, padding=2)
    
#     # 测试输入验证
#     print("\nTesting input validation...")
#     validate_input(input_image, proj_data, name="Test")
