from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CatsAndDogs(Dataset):
    def __init__(self, transforms=None):
        # stuff
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        data = # Some data read from a file or image
        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have
        
        
if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    custom_dataset = MyCustomDataset(..., transformations)