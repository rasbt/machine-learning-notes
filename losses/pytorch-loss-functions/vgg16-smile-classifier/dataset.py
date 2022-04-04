from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           download=True):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()
        
    def get_smile(attr):
        return attr[31]

    train_dataset = datasets.CelebA(root='.',
                                    split='train',
                                    transform=train_transforms,
                                    target_type='attr',
                                    target_transform=get_smile,
                                    download=download)

    valid_dataset = datasets.CelebA(root='.',
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_smile,
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='.',
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_smile,
                                   transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader
