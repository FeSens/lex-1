import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class VectorDataset(Dataset):
    def __init__(self, data_dir, data_file, batch_size):
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size

    @property
    def data(self):
        data = np.memmap(os.path.join(self.data_dir, self.data_file), dtype=np.uint16, mode='r')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(len(self.data) - self.batch_size, (1,)).item()
        x = torch.from_numpy((self.data[ix:ix+self.batch_size]).astype(np.int64))
        y = torch.from_numpy((self.data[ix+1:ix+1+self.batch_size]).astype(np.int64))
        return x, y
