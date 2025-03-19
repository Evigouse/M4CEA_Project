import torch
from tqdm import tqdm
from Finetune.finetune_utils import classcification_criterion
from torch.nn.utils import clip_grad_norm_
from torch import nn
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score, fbeta_score, accuracy_score, recall_score, precision_score

def get_predictions(logits):
    _, predicted = torch.max(logits, 1)
    return predicted

def train_epoch(cfg, epoch, data_loader, model, step_count, steps_per_update, optimizer,  writer):
    print("\n# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    running_loss = 0.0
    running_corrects = 0.0

    for sample,label in tqdm(data_loader):
        optimizer.zero_grad()
        step_count += 1
        sample = sample.to(device)
        label =  label.to(device)
        label = label.squeeze(1)

        if cfg.model == 'M4CEA':
            if cfg.task == 'chzu_onset_type':
                out = model(sample, fs=1000, mask=None, need_mask=False, task=cfg.task)

        probs = nn.Softmax(dim=1)(out)
        preds = torch.max(probs, 1)[1]
        loss = classcification_criterion(cfg,out,label.long())
        loss.backward()
        running_loss += loss.item() * sample.size(0)
        running_corrects += torch.sum(preds == label.data)

        if step_count % steps_per_update == 0:
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    writer.add_scalar('train_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('train_acc_epoch', epoch_acc, epoch)
    print(f'\nIn epoch {epoch}, training loss is {epoch_loss}, training acc is {epoch_acc}.')

    return epoch_loss

def val_epoch(cfg, epoch, data_loader, model, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    for sample, label in tqdm(data_loader):
        sample = sample.to(device)
        label = label.to(device)
        label = label.squeeze(1)
        with torch.no_grad():
            if cfg.model == 'M4CEA':
                if cfg.task == 'chzu_onset_type':
                    out = model(sample,fs=1000, mask=None,need_mask=False,task=cfg.task)

        probs = nn.Softmax(dim=1)(out)
        preds = torch.max(probs, 1)[1]
        loss = classcification_criterion(cfg,out, label.long())
        running_loss += loss.item() * sample.size(0)
        running_corrects += torch.sum(preds == label.data)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    writer.add_scalar('val_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('val_acc_epoch', epoch_acc, epoch)
    print(f'\nIn epoch {epoch}, validation loss is {epoch_loss}, validation acc is {epoch_acc}.')
    return epoch_loss