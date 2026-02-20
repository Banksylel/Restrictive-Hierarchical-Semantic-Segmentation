import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from Data.dataset import SegDataset

# splits the image ids into train, test and val sets. also gets the indices of the images
def split_ids(len_ids, val_img, test_img, test_remove):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))
    
    if val_img == 'None':
        train_indices, val_indices = train_test_split(
            np.linspace(0, len_ids - 1, len_ids).astype("int"),
            test_size=val_size,
            random_state=42,
        )
    else:
        train_indices = np.linspace(0, len_ids - 1, len_ids).astype("int")
        val_indices = np.linspace(0, len(val_img) - 1, len(val_img)).astype("int")
    
        
    if test_remove == False:
        if test_img == 'None':
            train_indices, test_indices = train_test_split(
                train_indices, test_size=test_size, random_state=42
            )
        else:
            train_indices = np.linspace(0, len_ids - 1, len_ids).astype("int")
            val_indices = np.linspace(0, len(test_img) - 1, len(test_img)).astype("int")
    else:
        test_indices = 'Empty'

    return train_indices, test_indices, val_indices


def get_dataloaders(input_paths, target_paths, class_tree, class_map, batch_size, val_batch_size='None', val_img='None', val_target='None', test_img='None', test_target='None', img_size='None', test_remove='None', types='None', workers_num='None', num_classes=None, model_type=None):
    if workers_num == 'None' or workers_num == -1:
        workers = multiprocessing.Pool()._processes
    else:
        workers = workers_num

    # augmentation, resize and normalisation for train
    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=False),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # resize and normalisation for test and val
    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=False),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # resize and grayscale for target 
    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((img_size, img_size)), transforms.Grayscale()]
    )

    # if predict mode, train dataset only
    if types=='Predict':
        # process train DS
        train_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            clss_t=class_tree,
            clss_m=class_map,
            transform_input=transform_input4test,
            transform_target=transform_target,
            model_type=model_type,
        )
        train_indices = np.linspace(0, len(input_paths) - 1, len(input_paths)).astype("int")
        train_dataset = data.Subset(train_dataset, train_indices)
        # load train DS
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        return 'Empty', train_dataloader, 'Empty'
    # otherwise train, test and val 
    else:
        # augmentation
        # process train DS. applys the agumentation/rezise/normalisation to images and masks
        train_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            clss_t=class_tree,
            clss_m=class_map,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=True,
            vflip=False,
            affine=True,
            classes=num_classes,
            model_type=model_type,
        )
        # process test DS. applys the agumentation/rezise/normalisation to images and masks
        if test_remove == False:
            if test_target == 'None':
                test_dataset = SegDataset(
                    input_paths=input_paths,
                    target_paths=target_paths,
                    clss_t=class_tree,
                    clss_m=class_map,
                    transform_input=transform_input4test,
                    transform_target=transform_target,
                    classes=num_classes,
                    model_type=model_type,
                )
            else:
                test_dataset = SegDataset(
                    input_paths=test_img,
                    target_paths=test_target,
                    clss_t=class_tree,
                    clss_m=class_map,
                    transform_input=transform_input4test,
                    transform_target=transform_target,
                    classes=num_classes,
                    model_type=model_type,
                )
        
        # process val DS. applys the agumentation/rezise/normalisation to images and masks
        if val_target == 'None':
            val_dataset = SegDataset(
                input_paths=input_paths,
                target_paths=target_paths,
                clss_t=class_tree,
                clss_m=class_map,
                transform_input=transform_input4test,
                transform_target=transform_target,
                classes=num_classes,
                model_type=model_type,
            )
        else:
            val_dataset = SegDataset(
                input_paths=val_img,
                target_paths=val_target,
                clss_t=class_tree,
                clss_m=class_map,
                transform_input=transform_input4test,
                transform_target=transform_target,
                classes=num_classes,
                model_type=model_type,
            )
    
    # split DS into train, test and val indices (image id number)
    train_indices, test_indices, val_indices = split_ids(len(input_paths), val_img, test_img, test_remove)
    
    # splits data into subset if internal splitting is turned on
    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)
    if test_remove == False:
        test_dataset = data.Subset(test_dataset, test_indices)
    

    # loads train DS into dataloaders
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
    )
    # loads test DS into dataloaders
    if test_remove == False:
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
        )
    else:
        test_dataloader = 'Empty'
    # loads val DS into dataloaders
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
    )

    return train_dataloader, test_dataloader, val_dataloader



