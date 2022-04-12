"""
    This part contains a data pre-processing, (From MNIST10)
    which are normalization and conversion to the pytorch 'Tensor' type.
    Doing Image Augmentation
    For the normalization, mean = 0.1307 and std = 0.3081 were used, which values are computed on the whole training set.
"""

import torch
#For random number carrier
import numpy as np
import torch.utils.data as data
#For Augmendation the image/ input
import torchvision.transforms as transforms
#For loading data from MNIST
import torchvision.datasets as datasets
#For sampling the dataset
from torch.utils.data.sampler import SubsetRandomSampler

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

#data loader for train set, validation set
# value extracted from entire training set.
def get_mnist_train_valid_loader(data_dir, # data saved direction
                                 batch_size,
                                 random_seed,
                                 valid_size=0.1, # validation size / traning size (Usually 0.25)
                                 shuffle=True,
                                 show_sample=False,
                                 num_workers=4,#Use 4 cpu for dataloader
                                 pin_memory=False): #do not directly use pin_memory(dataloader)

    #normalize : mean = 0.1307 and std = 0.3081 :: hyperparameter?
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transforms
    #Compose : Composes several transforms together.  
    valid_transform = transforms.Compose([
        #Change the size
        transforms.Resize((32,32)),
        #ToTensor() : PIL Image or NumPy -> FloatTensor
        transforms.ToTensor(),
        #Defined Normalize
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalize,
    ])

    # download the dataset / set transfrom!
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
        #download = True : download MNIST file -> add data file in this folder
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset) # train set length tuple
    indices = list(range(num_train)) # [0~train set-1] list
    split = int(np.floor(valid_size * num_train)) # valid set size = split size

    if shuffle:
        np.random.seed(random_seed) #Use seed_number from parameter
        np.random.shuffle(indices) # suffle [0~train set-1] list

    #train_idx : split~ / valid_idx : 0~split-1 
    train_idx, valid_idx = indices[split:], indices[:split]
    #Sample elements' index randomly from a given list w/o replacement
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    #DataLoader -> save my dataset from full dataset & random index sample
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        raise NotImplementedError
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuflle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        image, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    #Return loader data
    return (train_loader, valid_loader)

#data loader for test set
def get_mnist_test_loader(data_dir,
                          batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=False):
    
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    #define transform
    #test set also has Resize/ToTensor/Normalize Augmentation
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,                                         # We can load test_dataset using argument 'train=False'
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


# CIFAR10 dataloader function
def get_cifar10_100_train_valid_loader(dataset,
                                       data_dir,
                                       batch_size,
                                       augment,
                                       random_seed,
                                       valid_size=0.1,
                                       shuffle=True,
                                       num_workers=4,
                                       distributed=False,
                                       pin_memory=False):
    
    assert ((valid_size >= 0) and (valid_size <= 1)), \
        "[!] valid_size should be in the range [0, 1]."

    # value extracted from entire training set.
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010], # Each channels has mean, std respectively
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    #augmentation for valid set
    if augment:
        if dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4), #crop the size
                transforms.RandomHorizontalFlip(),    #FlipHorizontaly
                transforms.ToTensor(),                #Make them as a tensor
                normalize,                            #Normalize by using pre-defined normalize variable
            ])
        elif dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.RandomRotation((-15,15)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        ) #download = True .. same as MNIST

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn
    )

    if not valid_idx:
        valid_sampler = None
        valid_loader = None
    else:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return (train_loader, train_sampler, valid_loader)


def get_cifar10_100_test_loader(dataset,
                                data_dir,
                                batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    #define transform : Do not augemt for test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
    elif dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader