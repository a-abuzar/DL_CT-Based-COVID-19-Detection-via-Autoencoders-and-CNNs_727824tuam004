import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CTScanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images (should have COVID and non-COVID subdirs).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        self.covid_paths = glob.glob(os.path.join(data_dir, 'COVID', '*.*'))
        self.non_covid_paths = glob.glob(os.path.join(data_dir, 'non-COVID', '*.*'))
        
        # 1 for COVID, 0 for non-COVID
        self.image_paths = self.covid_paths + self.non_covid_paths
        self.labels = [1] * len(self.covid_paths) + [0] * len(self.non_covid_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir, batch_size=32, img_size=(224, 224), split_ratios=(0.7, 0.15, 0.15)):
    """Returns train, val, and test dataloaders."""
    # Data transformations
    # For training: Resize, augmentations, to tensor
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), # Converts to [0, 1] range
        # Normalize assuming grayscale
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    # For val/test: Only resize and to tensor
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = CTScanDataset(data_dir, transform=train_transform) # Base dataset
    
    # Calculate lengths
    total_len = len(full_dataset)
    train_len = int(split_ratios[0] * total_len)
    val_len = int(split_ratios[1] * total_len)
    test_len = total_len - train_len - val_len
    
    # Ensure there is data
    if total_len == 0:
        raise ValueError(f"No images found in {data_dir}. Please run src/create_mock_data.py first.")

    # Split dataset
    import torch
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update transforms for val and test
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the pipeline
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
        images, labels = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
    except ValueError as e:
        print(e)
