import csv
import glob
import os
from collections import Counter
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from data.transforms import make_transform_composition
from data.utils.read_data import read_image_paths_in_folder
from data.utils.split_train_val import split_train_val


class CatDataloader(data.Dataset):
    """
    Dataloader to load the traffic signs.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.single_object_name_to_one_hot = {'sajt': [1., 0.], 'pali': [0., 1.],
                                              'mindket_cica': [1., 1.], 'egyik_sem': [0., 0.]}
        self.object_name_to_one_hot = {'sajt': [1., 0., 0.], 'pali': [0., 1., 0.],
                                       'mindket_cica': [0., 0., 1.], 'egyik_sem': [0., 0., 0.]}
        self.original_paths, self.original_labels, self.labels_names = self.load_labels_and_paths()

        # undersample the overpresented, and oversample the underpresented classes
        if self.split == 'train' and self.config.balanced_classes:
            self.class_sample_numbers = self.dataset_balance_class_probabilities()
            self.sample_classes()
        else:
            self.paths, self.labels = self.original_paths, self.original_labels

        assert self.__len__(), f'No data were found in {self.config.dataset_path}.'

        self.transforms = make_transform_composition(self.config.transforms, self.split)

    def __getitem__(self, item):
        # load input image
        path = self.paths[item]
        with Image.open(path) as im:
            im = im.convert('RGB')
            input_image = self.transforms(im)

        # use the labels if we have them (during train and val) otherwise use a [] placeholder
        if self.labels is not None:
            label = int(self.labels[item])
            if self.config.multilabel_classification:
                if self.config.single_object_classes:
                    label_one_hot = self.single_object_name_to_one_hot[self.labels_names[label]]
                else:
                    label_one_hot = self.object_name_to_one_hot[self.labels_names[label]]
            else:
                label_one_hot = (np.arange(self.config.num_classes) == label).astype(float)
            label_one_hot = torch.as_tensor(label_one_hot, dtype=torch.float)
            label = torch.as_tensor(label, dtype=torch.long)
        else:
            label = []
            label_one_hot = []

        return {'paths': path, 'input_images': input_image, 'labels': label,
                'labels_one_hot': label_one_hot}

    def __len__(self):
        return len(self.paths)

    def load_labels_and_paths(self):
        """
        Load data paths and labels according to the current split (train/val)
        from the corresponding csv file in the dataset path (from the config).
        During training if the csv file does not exist make one for the train/val splits.
        During eval if the csv file does not exist use the whole dataset for evaluation.

        :return: [paths, labels]: list of the paths and list of the corresponding labels
        """
        filename = self.split + '_w_negs.csv' if self.config.include_negatives else self.split + '.csv'
        split_file_path = os.path.join(self.config.dataset_path, filename)
        if not os.path.exists(split_file_path):
            if self.split == 'train':
                split_train_val(self.config.train_val_split, self.config.dataset_path, self.config.include_negatives)
            elif self.split == 'test':
                return read_image_paths_in_folder(self.config.dataset_path), None, self.get_class_names()
            else:
                raise ValueError(f'Mode should be `train` for train-val split or `test` for testing on images.')

        with open(split_file_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        paths, labels = list(zip(*data))

        return np.array(paths), np.array(labels, dtype=int), self.get_class_names()

    def get_class_names(self):
        with open('classnumber_classname.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        return list(zip(*data))[-1]

    def dataset_balance_class_probabilities(self):
        """
        Calculate how many elemenents to sample by class. Do not have more than 10x sample from one class.
        :return: dictionary of the sample numbers by class (either the mean of the sample numbers or 10xclass_size)
        """
        class_numbers = list(Counter(self.original_labels).keys())
        class_sizes = list(Counter(self.original_labels).values())
        # use only the object classes for calculating the class size mean
        try:
            egyik_sem_index = self.labels_names.index('egyik_sem')
            egyik_sem_class_index = class_numbers.index(egyik_sem_index)
            class_sizes_mean = np.mean(class_sizes[:egyik_sem_class_index]+class_sizes[egyik_sem_class_index+1:])
        except ValueError:  # if there is no `egyik_sem` in class_numbers, then we have to add 0 instead of it
            class_sizes_mean = np.mean(class_sizes)
            # complete class numbers and class_sizes with new numbers and zeros for empty classes
            if len(self.labels_names) > len(class_numbers):
                class_numbers = class_numbers + list(range(len(self.labels_names)))[len(class_numbers):]
                class_sizes = class_sizes + [0] * (len(class_numbers) - len(class_sizes))

        return {int(class_number): int(min(class_sizes_mean, class_size * 10))
                for class_number, class_size in zip(class_numbers, class_sizes)}

    def sample_classes(self):
        """
        Make the over or undersampling. This should be rerun after every epoch for
        new samples to use all the samples from the dataset.
        """
        if not self.config.balanced_classes:
            return

        new_paths = []
        new_labels = []

        for class_number in range(self.config.num_classes):
            class_mask = self.original_labels == class_number
            class_paths = self.original_paths[class_mask]
            new_paths.extend(np.random.choice(class_paths, size=self.class_sample_numbers[class_number]))
            new_labels.extend([class_number] * self.class_sample_numbers[class_number])

        self.paths, self.labels = new_paths, new_labels
