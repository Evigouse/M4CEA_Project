import os
from tensorboardX import SummaryWriter
from Finetune.finetune_config import cfg
from Finetune.finetune_utils import PrepareSaveFile, EarlyStopping, PrepareDataLoader, PrepareModel, PrepareOptimizer, get_scheduler, leftBestModel,adjust_learning_rate
from Pretrain.random_seed import setup_seed
from Finetune.finetune_train_eval import train_epoch,val_epoch
import pickle

def main(cfg, train=False):
    setup_seed(cfg.seed)
    data_loader = PrepareDataLoader(cfg, train=train)
    model = PrepareModel(cfg, train=train)
    if train:
        savename=cfg.model
        ckpt_path, logp,  loss_path = PrepareSaveFile(cfg)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
        optimizer = PrepareOptimizer(cfg, model)
        writer = SummaryWriter(logdir=logp)
        load_batch_size = min(cfg.max_device_batch_size, cfg.batch_size)
        assert cfg.batch_size % load_batch_size == 0
        steps_per_update = cfg.batch_size // load_batch_size
        optimizer.zero_grad()
        step_count = 0
        train_epoch_losses = {}
        val_epoch_losses = {}
    for i in range(1, cfg.epoch + 1 if train else 2):
        if train:
            adjust_learning_rate(cfg, optimizer, i)
            epoch_loss = train_epoch(cfg, i, data_loader[0], model, step_count, steps_per_update, optimizer,  writer=writer)
            train_epoch_losses[i] = epoch_loss
            val_loss = val_epoch(cfg, i, data_loader[1], model, writer=writer)
            val_epoch_losses[i] = val_loss
            save_file_path = os.path.join(ckpt_path, cfg.task+'_'+'save_{}.pth.tar'.format(i))
            earlystopping(val_loss, model, save_file_path, i, optimizer)
            if earlystopping.early_stop:
                writer.close()
                print('earlystopping!')
                break
            leftBestModel(ckpt_path)
        else:
            evaluate(cfg, data_loader, model)

    if train:
        writer.close()
        with open(os.path.join(loss_path,'total_epoch_train_'+ savename+'_'+str(i) + '.pkl'), 'wb') as f:
            pickle.dump(train_epoch_losses, f)
        with open(os.path.join(loss_path,'total_epoch_val_'+ savename+'_'+ str(i) + '.pkl'), 'wb') as f:
            pickle.dump(val_epoch_losses, f)

if __name__ == "__main__":
    train = False
    if train == True:
        seed_list = [42, 2024, 3407, 4399, 114514]
        for i in [0,1,2, 3,4]:
            a=cfg(seed=seed_list[i])
            main(cfg(seed=seed_list[i]), train=train)
    else:
        seed_list=[42, 2024, 3407, 4399, 114514]
        main(cfg(seed=seed_list[4]), train=train)