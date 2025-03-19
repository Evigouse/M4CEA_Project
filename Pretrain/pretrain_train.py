import torch
from tqdm import tqdm
from Pretrain.pretrain_utils import regression_criterion,probability_proportional_sampling
from torch.nn.utils import clip_grad_norm_
import random
from Pretrain.random_seed import setup_seed
from Pretrain.pretrain_config import cfg

def train_epoch(cfg, epoch, data_loader, model, step_count, steps_per_update, optimizer,  writer):
    print("\n# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    running_loss = 0.0
    running_loss_angle = 0.0
    running_loss_magnitude = 0.0

    for sample,mask_weights in tqdm(data_loader):
        optimizer.zero_grad()
        step_count += 1
        sample = sample.to(device)
        mask = probability_proportional_sampling(mask_weights, 160)
        mask = torch.tensor(mask)
        mask = mask.to(device)
        rec_magnitude, rec_angle, magnitude, angle = model(sample,fs=1000,mask=mask,need_mask=True)

        loss_magnitude= regression_criterion(cfg,rec_magnitude, magnitude)
        loss_angle = regression_criterion(cfg,rec_angle, angle)
        loss = loss_magnitude + loss_angle
        loss.backward()

        running_loss += loss.item() * sample.size(0)
        running_loss_angle+= loss_angle.item() * sample.size(0)
        running_loss_magnitude+= loss_magnitude.item() * sample.size(0)

        if step_count % steps_per_update == 0:
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    loss_angle = running_loss_angle / len(data_loader.dataset)
    loss_magnitude = running_loss_magnitude / len(data_loader.dataset)

    writer.add_scalar('train_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('angle_loss_epoch', loss_angle, epoch)
    writer.add_scalar('magnitude_loss_epoch', loss_magnitude, epoch)


    print(f'\nIn epoch {epoch}, training loss is {epoch_loss}.')
    print(f'\nIn epoch {epoch}, angle loss is {loss_angle}.')
    print(f'\nIn epoch {epoch}, magnitude loss is {loss_magnitude}.')
    return epoch_loss, loss_angle, loss_magnitude