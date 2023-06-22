from torch.utils.data import Dataset


class BaseDetectionDataset(Dataset):
    def __init__(self):
        assert False, print('BaseDetectionDataset is not Unimplemented.')

    def __getitem__(self, item):
        pass
