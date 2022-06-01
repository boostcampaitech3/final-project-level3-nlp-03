import torch
import numpy as np
REAL_VALUE_DATA = ['korsts']
BIN_VALUE_DATA = []

class MultiSentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, data_type):
        self.encodings = encodings
        self.labels = labels
        if data_type in REAL_VALUE_DATA:
            self.labels = self.labels.astype(np.float32)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


class MultiSentDataset_STS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(torch.float)
        return item

    def __len__(self):
        return len(self.labels)