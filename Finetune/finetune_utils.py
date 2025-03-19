import torch
from Finetune.finetune_chzu_onset_type_dataloader import CHZU_onset_TypeLoader
from torch import optim, nn
from torch.utils.data import DataLoader
import os
import numpy as np
from Finetune.finetune_model import M4CEA
from functools import partial
import math

class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, path, epoch, optimizer):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch, optimizer):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, path)
        self.val_loss_min = val_loss

def PrepareDataLoader(cfg, train=True):
    class_mapping = {
        'CHZU_onset_TypeLoader':CHZU_onset_TypeLoader,
    }
    if cfg.task =='chzu_onset_type':
        loader_type = 'CHZU_onset_TypeLoader'
        root = 'Datapath'


    if loader_type in class_mapping:
        train_dataset = class_mapping[loader_type](root,split='train')
        valid_dataset = class_mapping[loader_type](root,split='val')
        test_dataset = class_mapping[loader_type](root,split='test')
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")

    if train:
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True)

        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)
    else:
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=cfg.test_batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=True)
    return (train_data_loader, valid_data_loader) if train else test_data_loader

def PrepareModel(cfg, train=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train:
        if cfg.model == 'M4CEA':
            if cfg.task == 'chzu_onset_type':
                model = M4CEA(in_chans=8, out_chans=8, pool='Avg', tie='sinusoidal',
                F1=8, kernel_length=64, inc=1, outc=8,kernel_size=(1, 63), pad=(0, 31), stride=1,bias=False, sample_len=200, alpha=2,
                in_dim=200, c_dim=32, seq_len=20, d_model=200,project_mode='linear', learnable_mask=True,
                embed_dim=200, depth=12, num_heads=8, mlp_ratio=4.,qkv_bias=False,qk_norm=partial(nn.LayerNorm, eps=1e-6), qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6),init_values=0.,window_size=None, attn_head_dim=None,type_num=4)

                model_dict = model.state_dict()
                print(cfg.pretrained_model_path)
                pretrained_dict = torch.load(cfg.pretrained_model_path)['state_dict']
                load_key, no_load_key, temp_dict = [], [], {}
                for k, v in pretrained_dict.items():
                    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                        temp_dict[k] = v
                        load_key.append(k)
                    else:
                        no_load_key.append(k)
                model_dict.update(temp_dict)
                model.load_state_dict(model_dict)

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
    return ckpt_path, logp,  loss_path

def leftBestModel(path):
    for i in os.listdir(path):
        if not i.endswith('.pth.tar'):
            os.rmdir(os.path.join(path, i))
    weights = os.listdir(path)
    if len(weights) == 1:
        return
    total = sorted(weights, key=lambda x: int(x.split('_')[-1][:-8]))
    for i in total[:-1]:
        os.remove(os.path.join(path, i))

def get_scheduler(cfg, optimizer):
    if cfg.scheduler_name == 'mul_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    elif cfg.scheduler_name == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-6, max_lr=1e-5, step_size_up=50, cycle_momentum=False)
    elif cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,gamma=0.1)
    else:
        raise NotImplementedError
    return scheduler

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

def classcification_criterion(cfg,output, label):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.loss_name =='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        loss = criterion(output, label)

    return loss