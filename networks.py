import numpy as np
import tinycudann as tcnn
import math
import torch
import torch.nn as nn
import json
# Helper functions for SIREN initialization
import torch.nn.functional as F
from labml_helpers.module import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, nhead, nhid, nlayers, output_dim):
        super(SimpleTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, nhid, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, src):
        src = src * math.sqrt(self.input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()

        self.w0 = w0

    def forward(self, input):
        return torch.sin(input * self.w0)



def sine_init(m, w0, num_input=None):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if num_input is None:
                num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1.0 / num_input, 1.0 / num_input)



############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda()
        else:
            raise NotImplementedError

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        # print(x.shape)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out
    



class FFB_encoder(nn.Module):
    def __init__(self, encoding_config, network_config, n_input_dims, bound=1.0, has_out=True):
        super().__init__()

        self.bound = bound # 设置bound属性，这通常用于归一化输入数据。

        ### The encoder part 设置网络的维度，并且在最前面添加输入维度。
        sin_dims = network_config["dims"] 
        sin_dims = [n_input_dims] + sin_dims
        self.num_sin_layers = len(sin_dims) # 计算网络层的数量。

        # 从编码配置中提取特征维度、基础分辨率和每层的缩放比例
        feat_dim = encoding_config["feat_dim"] #2
        base_resolution = encoding_config["base_resolution"] #2
        per_level_scale = encoding_config["per_level_scale"] #1.5

        assert self.num_sin_layers > 3, "The layer number (SIREN branch) should be greater than 3."
        ## 设置一个编码器，用于处理输入数据，使用哈希网格编码。
        grid_level = int(self.num_sin_layers - 2)
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": grid_level,
                "n_features_per_level": feat_dim,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.grid_level = grid_level
        print(f"Grid encoder levels: {grid_level}")

        self.feat_dim = feat_dim

        ### Create the ffn to map low-dim grid feats to map high-dim SIREN feats
        base_sigma = encoding_config["base_sigma"]
        exp_sigma = encoding_config["exp_sigma"]

        ffn_list = []
        for i in range(grid_level):
            ffn = torch.randn((feat_dim, sin_dims[2 + i]), requires_grad=True) * base_sigma * exp_sigma ** i

            ffn_list.append(ffn)

        self.ffn = nn.Parameter(torch.stack(ffn_list, dim=0))


        ### The low-frequency MLP part
        for layer in range(0, self.num_sin_layers - 1):
            setattr(self, "sin_lin" + str(layer), nn.Linear(sin_dims[layer], sin_dims[layer + 1]))

        self.sin_w0 = network_config["w0"]
        self.sin_activation = Sine(w0=self.sin_w0)
        self.init_siren()

        ### The output layers
        self.has_out = has_out
        if has_out:
            size_factor = network_config["size_factor"]
            self.out_dim = sin_dims[-1] * size_factor

            for layer in range(0, grid_level):
                setattr(self, "out_lin" + str(layer), nn.Linear(sin_dims[layer + 1], self.out_dim))

            self.sin_w0_high = network_config["w1"]
            self.init_siren_out()
            self.out_activation = Sine(w0=self.sin_w0_high)
        else:
            self.out_dim = sin_dims[-1] * grid_level


    ### Initialize the parameters of SIREN branch
    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))

            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin, w0=self.sin_w0)


    def init_siren_out(self):
        for layer in range(0, self.grid_level):
            lin = getattr(self, "out_lin" + str(layer))

            sine_init(lin, w0=self.sin_w0_high)


    def forward(self, in_pos):
        """
        in_pos: [N, 3], in [-bound, bound]

        in_pos (for grid features) should always be located in [0.0, 1.0]
        x (for SIREN branch) should always be located in [-1.0, 1.0]
        """
        # in_pos = in_pos.float()

        x = in_pos / self.bound								# to [-1, 1]
        in_pos = (in_pos + self.bound) / (2 * self.bound) 	# to [0, 1]

        grid_x = self.grid_encoder(in_pos)
        grid_x = grid_x.view(-1, self.grid_level, self.feat_dim)
        grid_x = grid_x.permute(1, 0, 2)

        embedding_list = []
        for i in range(self.grid_level):
            # print(grid_x[i].dtype, self.ffn[i].dtype)
            grid_output = torch.matmul(grid_x[i].float(), self.ffn[i])
            grid_output = torch.sin(2 * math.pi * grid_output)
            embedding_list.append(grid_output)

        if self.has_out:
            x_out = torch.zeros(x.shape[0], self.out_dim, device=in_pos.device)
        else:
            feat_list = []

        ### Grid encoding
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x)
            x = self.sin_activation(x)

            if layer > 0:
                x = embedding_list[layer-1] + x

                if self.has_out:
                    out_lin = getattr(self, "out_lin" + str(layer-1))
                    x_high = out_lin(x)
                    x_high = self.out_activation(x_high)

                    x_out = x_out + x_high
                else:
                    feat_list.append(x)

        if self.has_out:
            x = x_out
        else:
            x = feat_list

        return x
    

class NFFB(nn.Module):
    def __init__(self, config, out_dims=1):
        super().__init__()

        self.xyz_encoder = FFB_encoder(n_input_dims=2, encoding_config=config["encoding"],
                                       network_config=config["SIREN"], has_out=False)

        ### Initializing backbone part, to merge multi-scale grid features
        backbone_dims = config["Backbone"]["dims"]
        grid_feat_len = self.xyz_encoder.out_dim
        backbone_dims = [grid_feat_len] + backbone_dims + [out_dims]
        self.num_backbone_layers = len(backbone_dims)

        for layer in range(0, self.num_backbone_layers - 1):
            out_dim = backbone_dims[layer + 1]
            setattr(self, "backbone_lin" + str(layer), nn.Linear(backbone_dims[layer], out_dim))

        self.relu_activation = nn.ReLU(inplace=True)


    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]

        return params


    def forward(self, in_pos):
        """
        Inputs:
            x: (N, 2) xy in [-scale, scale]
        Outputs:
            out: (N, 1 or 3), the RGB values
        """

        # x = (in_pos - 0.5) * 2.0
        x = in_pos


        grid_x = self.xyz_encoder(x)
        out_feat = torch.cat(grid_x, dim=1)


        ### Backbone transformation
        for layer in range(0, self.num_backbone_layers - 1):
            backbone_lin = getattr(self, "backbone_lin" + str(layer))
            out_feat = backbone_lin(out_feat)

            if layer < self.num_backbone_layers - 2:
                out_feat = self.relu_activation(out_feat)

        out_feat = out_feat.clamp(-1.0, 1.0)

        return out_feat
    
# class NFFB(nn.Module):
#     def __init__(self, config, out_dims=1):
#         super().__init__()

#         self.xyz_encoder = FFB_encoder(n_input_dims=2, encoding_config=config["encoding"],
#                                        network_config=config["SIREN"], has_out=False)

#         ### Initializing backbone part with CNN to merge multi-scale grid features
#         grid_feat_len = self.xyz_encoder.out_dim
#         self.conv1 = nn.Conv1d(grid_feat_len, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(64, out_dims, kernel_size=3, stride=1, padding=1)

#         self.relu_activation = nn.ReLU(inplace=True)
#         self.out_dims = out_dims

#     @torch.no_grad()
#     # optimizer utils
#     def get_params(self, LR_schedulers):
#         params = [
#             {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
#         ]

#         return params

#     def forward(self, in_pos):
#         """
#         Inputs:
#             x: (N, 2) xy in [-scale, scale]
#         Outputs:
#             out: (N, 1 or 3), the RGB values
#         """

#         x = in_pos

#         grid_x = self.xyz_encoder(x)
#         out_feat = torch.cat(grid_x, dim=1).unsqueeze(2)  # Add channel dimension for Conv1d, and change unsqueeze(1) to unsqueeze(2)

#         ### Backbone transformation using CNN
#         out_feat = self.conv1(out_feat)
#         out_feat = self.relu_activation(out_feat)
#         out_feat = self.conv2(out_feat)
#         out_feat = self.relu_activation(out_feat)
#         out_feat = self.conv3(out_feat)

#         out_feat = out_feat.squeeze(2)  # Remove the channel dimension
#         out_feat = out_feat.clamp(-1.0, 1.0)

#         return out_feat



# class NFFB_att(nn.Module):
#     def __init__(self, config, out_dims=1):
#         super().__init__()
#         self.xyz_encoder = FFB_encoder(n_input_dims=2, encoding_config=config["encoding"],
#                                        network_config=config["SIREN"], has_out=False)

#         # 初始化多头注意力机制，这里设置了8个头，嵌入维度为128
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8)

#     def forward(self, x):
#         # 使用xyz_encoder获取特征
#         grid_x = self.xyz_encoder(x)

#         # 转换维度以匹配MultiheadAttention的输入要求(L, N, E)
#         query = grid_x[0].unsqueeze(1)  # (L, 1, E)
#         key = grid_x[1].unsqueeze(1)    # (L, 1, E)
#         value = grid_x[2].unsqueeze(1)  # (L, 1, E)

#         # 应用多头注意力机制
#         attn_output, _ = self.multihead_attn(query, key, value)

#         # 可能需要后续处理和整形
#         attn_output = attn_output.squeeze(1)  # 移除批维度，如果需要

#         return attn_output
    
# # 函数用于计算模型的参数数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # 提供的配置
# config = {
#     "network": {
#         "encoding": {
#             "feat_dim": 2,
#             "base_resolution": 96,
#             "per_level_scale": 1.5,
#             "base_sigma": 5.0,
#             "exp_sigma": 2.0,
#             "grid_embedding_std": 0.01
#         },
#         "SIREN": {
#             "dims" : [32, 32, 32, 32],
#             "w0": 30.0,
#             "w1": 30.0,
#             "size_factor": 1
#         },
#         "Backbone": {
#             "dims": [64, 64]
#         }
#     }
# }

# # 假设输出维度为 3 (例如 RGB 图像)
# out_dims = 1
# nffb_model = NFFB(config["network"], out_dims=1)

# # 实例化 NFFB 网络并转换到合适的数据类型（如Half）
# nffb_model = nffb_model.float().to('cuda' if torch.cuda.is_available() else 'cpu')
# print(nffb_model)

# x = torch.ones(16384, 2).cuda()

# y = nffb_model(x)
# print(y.shape)