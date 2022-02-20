from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler
import torch.utils.data as tdata
import numpy as np
import os
from PIL import Image
from torchtoolbox.transform import Cutout

def get_loader_ciafar_10(data_dir,valid_size,augmentation,batch_size,do_shuffle=True,number_workers=4, pin_memery=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if augmentation:
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            Cutout(),
            transforms.ToTensor(),
            normalize]
            )
    else:
        train_transform=transforms.Compose( [transforms.ToTensor(),
            normalize])
    
    valid_transform=transforms.Compose([transforms.ToTensor(),
            normalize]) 


    train_dataset=datasets.CIFAR10(root=data_dir, train=True, 
                                    download = False, transform=train_transform,)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=False,transform=valid_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=False,transform=valid_transform)

    num_data=len(train_dataset)
    index_list=list(range(num_data))
    if do_shuffle:
        np.random.shuffle(index_list)
    valid_num=int(np.floor(valid_size * num_data))
    train_index, valid_index=index_list[valid_num:], index_list[:valid_num]
    train_sampler, valid_sampler = SubsetRandomSampler(train_index),SubsetRandomSampler(valid_index)

    train_loader = tdata.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    sampler=train_sampler,num_workers = number_workers,pin_memory = pin_memery)
    valid_loader = tdata.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                    sampler=valid_sampler,num_workers = number_workers,pin_memory = pin_memery)
    test_loader = tdata.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    shuffle=False,num_workers = number_workers,pin_memory = pin_memery)
    return train_loader, valid_loader, test_loader 