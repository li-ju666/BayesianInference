from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


def load_data():
    transforms = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = CIFAR10(
        root='data', train=True, download=True, transform=transforms)
    test_data = CIFAR10(
        root='data', train=False, download=True, transform=transforms)

    return train_data, test_data


def main():
    train_data, test_data = load_data()
    print(len(train_data), len(test_data))

    # build dataloader
    trainloader = DataLoader(
        train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(
        test_data, batch_size=64, shuffle=False)
