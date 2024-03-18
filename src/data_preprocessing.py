import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define a custom dataset class for loading images and labels
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        age = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, age


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])