from __future__ import print_function, division

import pandas as pd
from torch.utils.data import Dataset

from .utils import getIds, getLabels, getImagesById


class ProteinDataset(Dataset):

	__colors__ = ['blue', 'green', 'red', 'yellow']

    def __init__(self, root_dir, csv_file=None, transform=None):
        self.root_dir = root_dir
        self.ids = getIds(self.root_dir)
        self.labels_frame = getLabels(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        images = getImagesById(self.root_dir, img_id)
        
