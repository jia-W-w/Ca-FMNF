import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
import numpy as np
from .mambairunet_arch import VSSBlock


class prj_fun(Function):
    @staticmethod
    def forward(self, image, options):
        self.save_for_backward(options)
        return ctlib.projection(image, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection_t(grad_output.contiguous(), options)
        return grad_input, None

class prj_t_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.projection_t(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection(grad_output.contiguous(), options)
        return grad_input, None

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        
    def forward(self, image, options):
        return prj_fun.apply(image, options)

class projector_t(nn.Module):
    def __init__(self):
        super(projector_t, self).__init__()
        
    def forward(self, proj, options):
        return prj_t_fun.apply(proj, options)

class fidelity_module(nn.Module):
    def __init__(self, options):
        super(fidelity_module, self).__init__()
        self.options = nn.Parameter(options, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.projector = projector()
        self.projector_t = projector_t()
        
    def forward(self, input_data, proj):
        temp = self.projector(input_data, self.options) - proj
        intervening_res = self.projector_t(temp, self.options)
        out = input_data - self.weight * intervening_res
        return out
    

class adj_weight(nn.Module):
    def __init__(self, k):
        super(adj_weight, self).__init__()
        self.k = k

    def forward(self, x):
        return ctlib.laplacian(x, self.k)

def img2patch(x, patch_size, stride):
    x_size = x.size()
    Ph = x_size[-2]-patch_size+1
    Pw = x_size[-1]-patch_size+1
    y = torch.empty(*x_size[:-2], Ph, Pw, patch_size, patch_size, device=x.device)
    for i in range(patch_size):
        for j in range(patch_size):
            y[...,i,j] = x[...,i:i+Ph,j:j+Ph]
    return y[...,::stride,::stride,:,:]

def patch2img(y, patch_size, stride, x_size):
    Ph = x_size[-2]-patch_size+1
    Pw = x_size[-1]-patch_size+1
    y_tmp = torch.zeros(*x_size[:-2], Ph, Pw, patch_size, patch_size, device=y.device)
    y_tmp[...,::stride,::stride,:,:] = y
    x = torch.zeros(*x_size, device=y.device)
    for i in range(patch_size):
        for j in range(patch_size):
            x[...,i:i+Ph,j:j+Ph] += y_tmp[...,i,j]
    return x

class img2patch_fun(Function):    

    @staticmethod
    def forward(self, x, size):
        self.save_for_backward(size)
        patch_size = size[0]
        stride = size[1]
        p_size = size[5:]
        y = img2patch(x, patch_size, stride)
        out = y.reshape(y.size(0), p_size[1]*p_size[2], p_size[3]*p_size[4])
        return out

    @staticmethod
    def backward(self, grad_output):
        size = self.saved_tensors[0]
        patch_size = size[0]
        stride = size[1]
        x_size = size[2:5]
        p_size = size[5:]
        y = grad_output.view(grad_output.size(0), *p_size)
        grad_input = patch2img(y, patch_size, stride, (grad_output.size(0), *x_size))
        return grad_input, None

class patch2img_fun(Function):

    @staticmethod
    def forward(self, x, size):
        self.save_for_backward(size)
        patch_size = size[0]
        stride = size[1]
        x_size = size[2:5]
        p_size = size[5:]
        y = x.view(x.size(0), *p_size)
        out = patch2img(y, patch_size, stride, (x.size(0), *x_size))
        return out

    @staticmethod
    def backward(self, grad_output):
        size = self.saved_tensors[0]
        patch_size = size[0]
        stride = size[1]
        p_size = size[5:]
        y = img2patch(grad_output, patch_size, stride)
        grad_input = y.reshape(grad_output.size(0), p_size[1]*p_size[2], p_size[3]*p_size[4])
        return grad_input, None

class Im2Patch(nn.Module):
    def __init__(self, patch_size, stride, img_size) -> None:
        super(Im2Patch, self).__init__()
        Ph = (img_size-patch_size) // stride + 1
        Pw = (img_size-patch_size) // stride + 1
        self.size = torch.LongTensor([patch_size, stride, 1, img_size, img_size, 1, Ph, Pw, patch_size, patch_size])

    def forward(self, x):
        return img2patch_fun.apply(x, self.size)

class Patch2Im(nn.Module):
    def __init__(self, patch_size, stride, img_size) -> None:
        super(Patch2Im, self).__init__()
        Ph = (img_size-patch_size) // stride + 1
        Pw = (img_size-patch_size) // stride + 1
        self.size = torch.LongTensor([patch_size, stride, 1, img_size, img_size, 1, Ph, Pw, patch_size, patch_size])
        m = torch.ones(1, Ph * Pw, patch_size ** 2)
        mask = patch2img_fun.apply(m, self.size)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        y = patch2img_fun.apply(x, self.size)
        out = y / self.mask
        return out

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

    def forward(self, x, adj):
        t = x.view(-1, x.size(2))
        support = torch.mm(t, self.weight)
        support = support.view(x.size(0), x.size(1), -1)
        out = torch.zeros_like(support)
        for i in range(x.size(0)):
            out[i] = torch.mm(adj[i], support[i])
        out = out + self.bias
        return out


class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        # self.block3 = GCN(p_size**2, gcn_hid_ch)
        # self.block4 = GCN(gcn_hid_ch, p_size**2)
        self.vss_block = VSSBlock(hidden_dim=hid_channels)

        self.image2patch = Im2Patch(p_size, stride, img_size)
        self.patch2image = Patch2Im(p_size, stride, img_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        patch = self.image2patch(input_data)
        # tmp3 = self.relu(self.block3(patch, adj))
        # tmp3 = self.block4(tmp3, adj)
        # print(tmp3.shape)
        # tmp3 = self.patch2image(tmp3)
        # print(tmp3.shape)
        tmp3 = self.vss_block(patch, (127, 127))
        tmp3 = self.patch2image(tmp3)
        output = tmp1  + tmp2 + tmp3
        output = self.relu(output)
        return output
    
class MAGIC(nn.Module):
    def __init__(self, options, block_num=50, hid_channels=64, kernel_size=5, padding=2, img_size=256, p_size=6, stride=2, gcn_hid_ch=64):
        super(MAGIC, self).__init__()
        self.blocks = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options) for i in range(block_num)])

        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_data, proj):
        x = input_data
        for block in self.blocks:
            x = block(x, proj)
        return x


    
    
# views = 20#Number of projection views
# dets = 624 #Number of detector elements
# width = 512 #Width of image
# height = 512 #Height of image
# dImg = 0.006641 #Physical size of a pixel
# dDet = 0.012858 #Physical size of a detector element
# Ang0 = 0.0 #Angle of the first projection
# # dAng = (120.0 / views) * (np.pi / 180.0)  # 将角度从度转换为弧度，有限角度
# dAng = 2 * np.pi/ views  # 调整角度间隔以覆盖360度，稀疏角度

# s2r = 5.95 #Distance between x-ray soruce and rotation cente
# d2r = 4.906 #Distance between rotation center and detector
# binshift = -0.0013 #Shift of detector
# scan_type = 0 #cannint type, 0: equal distance fan beam, 1: euqal angle fan beam, 2: parallel beam

# gcn_hid_ch=64
# hid_channels = 64
# kernel_size = 5
# padding = 2
# img_size = 512
# p_size = 8
# stride = 4
# options =  torch.tensor([views, dets, width, height, dImg, dDet, Ang0, 2 * np.pi / views, s2r, d2r, binshift, scan_type]).cuda()
# # 生成一些合成数据作为输入
# batch_size = 1  # 这里可以根据您的需求更改
# synthetic_image = torch.rand(batch_size, 1, 512, 512).cuda()
# # 用模型处理这些数据  # 随机生成的图像
# synthetic_proj = torch.rand(batch_size, 1, views, dets).cuda()
# # 用模型处理这些数据   # 随机生成的投影数据
# model = MAGIC(options, img_size=512, p_size=8, stride=4).cuda()
# # model = Iter_block(hid_channels, kernel_size, padding, img_size, p_size, stride, gcn_hid_ch, options).cuda()
# # 用模型处理这些数据
# output_image = model(synthetic_image, synthetic_proj)
# print(model)
# print(output_image.shape)