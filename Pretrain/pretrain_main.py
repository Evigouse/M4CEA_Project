import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorboardX import SummaryWriter
from Pretrain.pretrain_config import cfg
from Pretrain.pretrain_utils import PrepareSaveFile, Pretrain_PrepareDataLoader, Pretrain_PrepareModel, PrepareOptimizer, adjust_learning_rate
from Pretrain.random_seed import setup_seed
from Pretrain.pretrain_train import train_epoch
import pickle
import torch


def main(cfg, train=False):
    setup_seed(cfg.seed)
    data_loader = Pretrain_PrepareDataLoader(cfg, train=train)
    model = Pretrain_PrepareModel(cfg, train=train)
    if train:
        save_epoch=cfg.save_epoch
        savename=cfg.model
        ckpt_path, logp,  loss_path = PrepareSaveFile(cfg)
        optimizer = PrepareOptimizer(cfg, model)
        writer = SummaryWriter(logdir=logp)
        load_batch_size = min(cfg.max_device_batch_size, cfg.batch_size)
        assert cfg.batch_size % load_batch_size == 0
        steps_per_update = cfg.batch_size // load_batch_size
        optimizer.zero_grad()
        step_count = 0
        epoch_losses = {}
        losses_angle = {}
        losses_magnitude = {}
    for i in range(1, cfg.epoch + 1 if train else 2):
        if train:
            adjust_learning_rate(cfg, optimizer, i)
            epoch_loss, loss_angle, loss_magnitude = train_epoch(cfg, i, data_loader, model, step_count, steps_per_update, optimizer, writer=writer)
            epoch_losses[i] = epoch_loss
            losses_angle[i] = loss_angle
            losses_magnitude[i] = loss_magnitude
            if i % save_epoch == 0:
                torch.save({
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(ckpt_path,  savename + '_epoch_' + str(i) + '.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join(ckpt_path,  savename + '_epoch_' + str(i) + '.pth.tar')))
        else:
            evaluate(cfg, data_loader, model)

    if train:
        writer.close()
        with open(os.path.join(loss_path,'total_epoch_'+ str(i) + '.pkl'), 'wb') as f:
            pickle.dump(epoch_losses, f)
        print(epoch_losses)

        with open(os.path.join(loss_path,'losses_angle_'+ str(i) + '.pkl'), 'wb') as f:
            pickle.dump(losses_angle, f)
        print(losses_angle)

        with open(os.path.join(loss_path,'losses_magnitude_'+ str(i) + '.pkl'), 'wb') as f:
            pickle.dump(losses_magnitude, f)
        print(losses_magnitude)

if __name__ == "__main__":
    train = True
    main(cfg, train=train)