class cfg:
    def __init__(self,seed):
        self.seed = seed
        self.name = cfg.model + '_' + cfg.task + '_' + str(seed)+'_' + cfg.type
        self.save_root = cfg.save_root+'_' + self.name + '_'

    model = 'M4CEA'
    save_root = ".\\result_finetune"
    type = 'finetune'
    loss_name ='CrossEntropyLoss'
    optimizer_name = 'AdamW'
    scheduler_name = 'step'
    max_device_batch_size=8
    batch_size = 64
    num_workers = 4
    epoch = 100
    warmup_epoch = 5
    patience = 30
    save_epoch=2
    learning_rate = 0.001
    min_lr=0.
    wd = 1e-4
    momentum = 0.9
    pretrained_model_path ='pretrained_model_path'
    task = 'chzu_onset_type'
    savelog = 'logs'
    saveweight = 'weights'
    saveloss ='finetune_loss'