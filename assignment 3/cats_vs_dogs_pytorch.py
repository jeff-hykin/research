import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from cats_vs_dogs_dataset import CatsAndDogs


class MnistNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        number_of_colors_in_picture = 3
        self.image_dimensions = (150, 150)
        # Create the layers we're going to use
        self.conv1 = nn.Conv2d(in_channels=number_of_colors_in_picture, out_channels=20, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5)
        self.fully_connected_layer1 = nn.Linear(in_features=30 * 5 * 5, out_features=300)
        self.fully_connected_layer2 = nn.Linear(in_features=300, out_features=75)
        self.fully_connected_layer3 = nn.Linear(in_features=75, out_features=10)

    def forward(self, incoming_data):
        #
        # Connect all the layers together
        #
        running_data = F.relu(self.conv1(incoming_data))
        # dimensions = (20, 28 , 28)
        running_data = self.pool(running_data)
        # dimensions = (20, 14 , 14)
        running_data = F.relu(self.conv2(running_data))
        # dimensions = (30, 10 , 10)
        running_data = self.pool(running_data)
        # dimensions = (30,  5 ,  5)
        running_data = running_data.view(-1, 5 * 5 * 30)
        # dimensions = (750)
        running_data = F.relu(self.fully_connected_layer1(running_data))
        # dimensions = (300)
        running_data = F.relu(self.fully_connected_layer2(running_data))
        # dimensions = (75)
        running_data = torch.sigmoid(self.fully_connected_layer3(running_data))
        # dimensions = (10)
        return running_data

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum'
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )

def train_and_test_loaders(batch_size, test_batch_size, data_loader_kwargs):
    # 
    # Pick what transformations to perform on the images
    # 
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    
    # 
    # Get The Datasets
    #
    where_to_save_dataset = '../data'
    # training_dataset = datasets.MNIST(where_to_save_dataset, train=True, download=True, transform=transformations)
    # test_dataset     = datasets.MNIST(where_to_save_dataset, train=False,               transform=transformations)
    training_dataset = CatsAndDogs(train_validate_or_test="train", transform=transformations)
    test_dataset     = CatsAndDogs(train_validate_or_test="test" , transform=transformations)
    
    # 
    # Create the loaders (feeds batches of data into model)
    # 
    train_loader = torch.utils.data.DataLoader( training_dataset, batch_size = batch_size     , shuffle = True, **data_loader_kwargs )
    test_loader  = torch.utils.data.DataLoader( test_dataset    , batch_size = test_batch_size, shuffle = True, **data_loader_kwargs )
    return train_loader, test_loader

def main():
    # 
    # Knobs
    # 
    batch_size      = 64
    test_batch_size = 1000
    epochs          = 10
    lr              = 0.01 # learning rate
    momentum        = 0.05
    use_cuda        = True
    seed            = 1 # random seed
    log_interval    = 10
    save_model      = True
    
    # add seed
    torch.manual_seed(seed)
    
    # 
    # Cuda or no cuda
    # 
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        data_loader_kwargs = {}
        device = torch.device("cpu")
    
    # 
    # get the data
    # 
    train_loader, test_loader = train_and_test_loaders(batch_size, test_batch_size, data_loader_kwargs)
    
    # 
    # Create the model
    # 
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum
    )
    
    # 
    # Train and test the model
    # 
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # 
    # Save model
    # 
    if (save_model):
        torch.save(model.state_dict(), "cats_and_dogs_cnn.pt")


if __name__ == '__main__':
    main()
