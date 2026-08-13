import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

class DIV2KDataset(Dataset):
    def __init__(self, img_dir="./data/div2k_samples", train=True, num_imgs = -1, crop_size=512, pad = 128):
        """
        Args:
            img_dir (str): Path to the image directory.
            train (bool): Whether the dataset is for training or testing.
        """
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.train = train
        self.num_imgs = num_imgs
        self.crop_size = crop_size
        self.pad = pad 
        if num_imgs > -1:
            self.img_files = self.img_files[:num_imgs]
        
        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.Resize(512),         # 
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(crop_size), 
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.train:
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)
            
        pad = self.pad
        if pad > 0:
            image = F.pad(image, (pad, pad, pad, pad), mode="constant", value=0.0)
        
        return image, self.img_files[idx], pad

# Set up the data loader
def get_dataloader(img_dir="./data/div2k_samples", batch_size=1, train=False, shuffle=False, num_imgs = -1, **kwargs):
    dataset = DIV2KDataset(img_dir, train=train, num_imgs = num_imgs,
                            crop_size=kwargs.get('crop_size', 512),
                            pad=kwargs.get('pad', 128))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if train else False)
