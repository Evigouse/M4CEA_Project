import numpy as np
import pickle
from MFDE import MFDE
import os
from einops import rearrange

def extract(data_path,type,save):
    path = os.path.join(data_path,type)
    save_path =os.path.join(save,type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for data in sorted(os.listdir(path)):
        data_name=data.split('.')[0]
        data_path = os.path.join(path, data)
        data = np.load(data_path).astype("float32")
        ex_data = data.copy()
        ex_data = ex_data.reshape(16, 20, -1)
        ex_data = rearrange(ex_data, 'N A T -> (N A) T')

        en = []
        for i in range(ex_data.shape[0]):
            temp = ex_data[i, :]
            out = MFDE(temp, 2, 6, 1, 5)
            avg_out = np.mean(out)
            en.append(avg_out)
        normalized_en = en / np.sum(en)

        save_en_folder = os.path.join(save_path, data_name + '_' + type + ".pkl")
        pickle.dump(
            {"data": data, "mfde": normalized_en},
            open(save_en_folder, "wb"), )