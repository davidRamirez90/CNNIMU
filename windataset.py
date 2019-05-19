import os
import glob
import pickle as cp
from torch.utils.data import Dataset



class windowDataSet(Dataset):
    """ Self window dataset definition """

    def __init__(self, dir, transform=None):
        """
            @:arg dir (string): path to dataset directory
            @:arg transform (callable, opt): Optional transform for samples
        """

        self.dir = dir
        self.transform = transform

    def __len__(self):
        files = [f for f in glob.glob(self.dir+'/*.pkl')]
        return len(files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, 'seq_{0:06}.pkl'.format(idx))
        file = open(img_path, 'rb')
        window = cp.load(file)

        if self.transform:
            window = self.transform(window)

        return window