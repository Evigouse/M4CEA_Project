import torch
import numpy as np
import os
import pickle

class Pretrain_Loader(torch.utils.data.Dataset):
    def __init__(self, root_chzu):
        self.root_chzu = root_chzu
        self.chzu_fnames = []
        self.choose_fnames = []

        for name in sorted(os.listdir(self.root_chzu)):
            self.chzu_fnames.append(os.path.join(self.root_chzu, name))

        self.choose_fnames = self.chzu_fnames

    def __len__(self):
        return len(self.choose_fnames)

    def chzu_load(self, index):
        data = pickle.load(open(self.choose_fnames[index], "rb"))
        samples=data['data']
        mask=data['mfde']
        samples = samples / (
                np.quantile(
                    np.abs(samples), q=0.95, axis=-1, keepdims=True
                )
                + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, mask

    def __getitem__(self, index):
        return self.chzu_load(index)