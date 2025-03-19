import math
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from functools import partial
import random
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trunc_normal_c(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_c(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_c(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TIE_Layer(nn.Module):
    def __init__(self, in_chans=8, out_chans=8, pool='Avg', tie='sinusoidal', F1=8, kernel_length=64, inc=1, outc=8,
                 kernel_size=(1, 63), pad=(0, 31), stride=1, bias=False, sample_len=200, alpha=2):
        self.pool = pool
        super(TIE_Layer, self).__init__()
        self.TIE = tie
        self.conv2DOutput = nn.Conv2d(1, F1, (1, kernel_length - 1), padding=(0, kernel_length // 2 - 1),
                                      bias=False)  # 'same'
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu3 = nn.GELU()
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm4 = nn.GroupNorm(4, out_chans)
        self.gelu4 = nn.GELU()
        self.test_feature=[]

        self.alpha = alpha
        if self.pool == 'Avg':
            self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
        elif self.pool == 'Max':
            self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.pooling = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=pad,
                                     stride=stride, bias=bias)
            self.weight_mat = np.ones(shape=[outc, inc, self.pooling.kernel_size[0], self.pooling.kernel_size[1]])
            self.pooling.weight = nn.Parameter(torch.Tensor(self.weight_mat), requires_grad=False)


        if self.TIE == 'linear':
            self.scale_factor = nn.Parameter(torch.zeros(1, ), requires_grad=True)
            self.scale_factor.requires_grad_ = True

        elif self.TIE == 'sinusoidal':
            self.scale_factor = nn.Parameter(torch.randn(sample_len), requires_grad=True)  # .to(device)
            self.scale_factor.requires_grad_ = True

        else:
            self.scale_factor = 0

        self.apply(_init_weights)

    def forward(self, data, fs):
        bs, ele_ch, sample_point = data.shape
        x1 = data.reshape(bs, ele_ch, sample_point * 5// fs , -1)
        y = rearrange(x1, 'B N A T -> B (N A) T')
        B, NA, T = y.shape
        x = y.unsqueeze(1)

        try:
            x = torch.add(self.conv2DOutput(x),
                          torch.mul(self.pooling(x),
                                    torch.reshape(self.scale_factor, [1, 1, 1, -1])))
        except:
            temp = torch.reshape(self.scale_factor, [1, 1, 1, -1]).to(device)
            x = torch.add(self.conv2DOutput(x),
                          torch.mul(self.pooling(x),
                                    temp))

        x = self.gelu1(self.norm1(x))
        x = self.gelu2(self.norm2(self.conv1(x)))
        x = self.gelu3(self.norm3(self.conv2(x)))
        x = self.gelu4(self.norm4(self.conv3(x)))
        x = rearrange(x, 'B C (N A) T -> B N A (T C)', N=ele_ch, A=sample_point * 5// fs )
        return x


class InputEmbedding(nn.Module):
    def __init__(self, in_dim=1000, c_dim=32, seq_len=4, d_model=1024, project_mode='linear', learnable_mask=True):
        super(InputEmbedding, self).__init__()

        self.mode = project_mode
        self.positional_encoding = nn.Parameter(torch.randn(seq_len * 16, d_model),
                                                requires_grad=True)  # learnable positional encoding
        if learnable_mask:
            self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)
        else:
            self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.softmax = nn.Softmax(dim=-1)
        if project_mode == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(1, c_dim, 150, 10),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
                nn.Conv1d(c_dim, c_dim * 2, 10, 5),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
            )

            self.cnn_proj = nn.Sequential(
                nn.Linear(c_dim * 2, d_model),
            )
        elif project_mode == 'linear':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
            )
        else:
            raise NotImplementedError

        self.apply(_init_weights)

    def forward(self, data, mask, need_mask, mask_by_ch, rand_mask, mask_len):

        batch_size, ch_num, seq_len, seg_len = data.shape
        if need_mask:
            masked_x = data.clone()
            if rand_mask:
                if mask_by_ch:
                    masked_x[:, mask[:, 0], mask[:, 1], :] = self.mask_encoding

                else:
                    masked_x = masked_x.reshape(batch_size, ch_num * seq_len, seg_len)
                    for i in range(masked_x.shape[0]):
                        masked_x[i, mask[i], :] = self.mask_encoding

            else:
                masked_x[:, :, -mask_len:, :] = self.mask_encoding
        else:
            masked_x = data.view(batch_size, ch_num * seq_len, seg_len)
        # projection
        if self.mode == 'cnn':
            masked_x = masked_x.view(batch_size * ch_num * seq_len, 1, seg_len)
            input_emb = torch.mean(self.cnn(masked_x), dim=-1)
            input_emb = self.cnn_proj(input_emb)
            input_emb = torch.transpose(input_emb, 1, 2)
        elif self.mode == 'linear':
            input_emb = self.proj(masked_x)  # (bat_size * ch_num, seq_len, d_model)

        # add encodings
        input_emb = input_emb + self.positional_encoding
        return input_emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim=512, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., init_values=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Transformer_Net(nn.Module):

    def __init__(self, embed_dim=512, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, window_size=None, attn_head_dim=None):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=window_size,
                attn_head_dim=attn_head_dim
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(_init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class M4CEA(nn.Module):
    def __init__(self, in_chans=8, out_chans=8, pool='Avg', tie='sinusoidal', F1=8, kernel_length=64, inc=1, outc=8,
                 kernel_size=(1, 63), pad=(0, 31), stride=1, bias=False, sample_len=1000, alpha=2,
                 in_dim=1000, c_dim=32, seq_len=4, d_model=512, project_mode='linear', learnable_mask=True,
                 embed_dim=512, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=0., window_size=None, attn_head_dim=None,
                 type_num=4):
        super(M4CEA, self).__init__()

        self.time = TIE_Layer(in_chans, out_chans, pool, tie, F1, kernel_length, inc, outc, kernel_size, pad, stride,
                              bias, sample_len, alpha)
        self.input_emb = InputEmbedding(in_dim, c_dim, seq_len, d_model, project_mode, learnable_mask)
        self.transformer = Transformer_Net(embed_dim, depth,
                                           num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate,
                                           drop_path_rate, norm_layer, init_values, window_size, attn_head_dim)
        self.typeclassifier = ClassificationHead(embed_dim, type_num)
        self.time_feature=[]
        self.transformer_feature=[]
        self.apply(_init_weights)

    def forward(self, data, fs, mask, need_mask=True, mask_by_ch=False, rand_mask=True, mask_len=None,task='huh_type'):
        x = self.time(data, fs)
        x = self.input_emb(x, mask, need_mask, mask_by_ch, rand_mask, mask_len)
        x = self.transformer(x)

        if task == 'chzu_onset_type':
            x = self.typeclassifier(x.mean(dim=1))
            return x


if __name__ == "__main__":
    import torch
    inputs = torch.rand(2, 16, 4000)

    net = M4CEA(in_chans=8, out_chans=8, pool='Avg', tie='sinusoidal', F1=8, kernel_length=64, inc=1, outc=8,
                kernel_size=(1, 63), pad=(0, 31), stride=1, bias=False, sample_len=200, alpha=2,
                in_dim=200, c_dim=32, seq_len=20, d_model=200, project_mode='linear', learnable_mask=True,
                embed_dim=200, depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False,
                qk_norm=partial(nn.LayerNorm, eps=1e-6), qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.,
                window_size=None, attn_head_dim=None)

    output= net(inputs, fs=1000, mask=None,need_mask=False,task='chzu_onset_type')
    print(output.shape)