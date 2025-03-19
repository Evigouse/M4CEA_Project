import torch
import numpy as np
import os
import pickle
import torchvision.transforms as transforms

class CHZU_onset_TypeLoader(torch.utils.data.Dataset):
    def __init__(self, root, sampling_rate=1000, split = 'train'):
        self.root = root
        self.default_rate = 1000
        self.sampling_rate = sampling_rate
        self.fnames, labels = [], []
        self.split = split

        for label in sorted(os.listdir(root)):
            for name in os.listdir(os.path.join(root, label)):
                self.fnames.append(os.path.join(root, label, name))

        index_temp = [i for i in range(len(self.fnames))]

        np.random.shuffle(index_temp)
        self.fnames_temp = []
        for c in index_temp:
            self.fnames_temp.append(self.fnames[index_temp[c]])

        self.list_fname = []
        if self.split =='train':
            self.list_fname = self.fnames_temp[0:int(0.6 * len(self.fnames_temp))]

        elif self.split =='val':
            self.list_fname = self.fnames_temp[int(0.6 * len(self.fnames_temp)):int(0.9 * len(self.fnames_temp))]

        elif self.split =='test':
            self.list_fname = self.fnames_temp[int(0.9 * len(self.fnames_temp)):len(self.fnames_temp)]


    def __len__(self):
        return len(self.list_fname)

    def __getitem__(self, index):
        data = pickle.load(open(self.list_fname[index], "rb"))
        X = data["selected_data"]
        sample_label = data['type_label']

        label_mapping = {'ABSZ': 0, 'FNSZ': 1, 'Spasm': 2, 'TCSZ': 3}
        label_encoder = transforms.Lambda(lambda x: label_mapping[x])
        self.label_array = np.array([label_encoder(sample_label)], dtype=int)

        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        X = torch.FloatTensor(X)
        return X, self.label_array