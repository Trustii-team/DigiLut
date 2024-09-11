from torch.utils.data import Dataset
from PIL import Image

class LungSet(Dataset):
    def __init__(self, dataframe, labels, transforms):
        self.X = dataframe
        self.y = labels.values
        self.transforms = transforms


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        df = self.X.iloc[idx]
        path = df['path']
        image = df['image']
        label = self.y[idx]
        return self.transforms(Image.open(path+image)), label