class cfg:
    seed =42
    model = 'M4CEA'
    loss_name ='MSE'
    optimizer_name = 'AdamW'

    max_device_batch_size=128
    batch_size = 256
    num_workers = 16
    epoch = 100
    save_epoch=20

    warmup_epoch = 30
    learning_rate = 0.001
    min_lr = 0.
    wd = 1e-4
    momentum = 0.9

    name = model + '_' + str(seed)
    type ='pretrain_learn_guide_mask'
    save_root = './result_pretrain' +'_'+ name+'_'+type+'_'
    savelog = 'logs'
    saveweight = 'weights'
    saveloss ='pretrain_loss'