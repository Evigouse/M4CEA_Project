import torch
from Pretrain.pretrain_dataloader_pro import Pretrain_Loader
from torch import optim, nn
from torch.utils.data import DataLoader
import os
import numpy as np
import math
from Pretrain.pretrain_model import M4CEA
from functools import partial
from einops import rearrange

def Pretrain_PrepareDataLoader(cfg, train=True):
    root = '../Database/Pretrain_Data'
    if train:
        train_dataset = Pretrain_Loader(root)
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True)

    return train_data_loader


def Pretrain_PrepareModel(cfg, train=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train:
        if cfg.model == 'M4CEA':
            model = M4CEA(in_chans=8, out_chans=8,pool='Avg', tie='sinusoidal', F1=8, kernel_length=64, inc=1, outc=8, kernel_size=(1,63), pad=(0,31), stride=1, bias=False, sample_len=200,alpha=2,
            in_dim=200, c_dim=32, seq_len=4*5, d_model=200, project_mode='linear', learnable_mask=True,
            embed_dim=200, depth=12,num_heads=8, mlp_ratio=4., qkv_bias=False, qk_norm=partial(nn.LayerNorm,eps=1e-6), qk_scale=None, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., window_size=None, attn_head_dim=None)

        else:
            raise NotImplementedError


    return model.to(device)

def PrepareSaveFile(cfg):
    i = 0
    while True:
        save_dir = cfg.save_root + str(i)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i += 1
    ckpt_path = os.path.join(save_dir, cfg.saveweight)
    logp = os.path.join(save_dir, cfg.savelog)
    loss_path = os.path.join(save_dir, cfg.saveloss)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(logp):
        os.makedirs(logp)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    return ckpt_path, logp, loss_path

def adjust_learning_rate(cfg, optimizer, epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    lr = cfg.learning_rate
    min_lr=cfg.min_lr
    if epoch < cfg.warmup_epoch:
        lr = lr * epoch / cfg.warmup_epoch
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - cfg.warmup_epoch) / (cfg.epoch - cfg.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def PrepareOptimizer(cfg, model):
    parameters = model.parameters()
    if cfg.optimizer_name =='SGD':
        optimizer = optim.SGD(parameters,
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.wd)
    elif cfg.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate * cfg.batch_size / 256,
                                  betas=(0.9, 0.999), weight_decay=cfg.wd)
    else:
        raise NotImplementedError
    return optimizer

def regression_criterion(cfg,rec, target):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.loss_name == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
        criterion.to(device)
        target = rearrange(target, 'b n a c -> b (n a) c')
        loss = criterion(rec, target)

    return loss

def probability_proportional_sampling(weights, num_samples):
    
    mask=[]
    for i in range(weights.shape[0]):
        sampled_indices = np.random.choice(len(weights[i,:]), size=num_samples, p=weights[i,:],replace=False)
        mask.append(sampled_indices)

    return mask