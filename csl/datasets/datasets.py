# -*- coding: utf-8 -*-
""" Datasets for the csl module

- CIFAR-10
- Fashion MNIST
- UCI's Adult
- ProPublica's COMPAS
- UTKFace

"""

import torch
import os, glob
import pandas as pd
import numpy as np
from PIL import Image


class CIFAR10:
    """CIFAR-10 dataset

    You can download the dataset in PyTorch tensor format from
    https://www.seas.upenn.edu/~luizf/data/cifar-10.zip

    ..warning:: For performance purposes, this class loads the full
                CIFAR-10 dataset to RAM. Even though it is less than 1 GB,
                you've been warned.

    Attributes
    ----------
    classes : list[str]
        Class labels
    train : bool
        True if training set or False otherwise.
    data : torch.tensor
        CIFAR-10 images.
    transform : callable
        Function applied to the data points before returning them.
    target : torch.tensor
        CIFAR-10 labels.
    target_transform : callable
        Function applied to the labels before returning them.

    Methods
    -------
    __len__()
        Return size of dataset.
    __get_item__()
        Return tuple (`torch.tensor`, `torch.tensor`) of images
        ([N] x [C = 3] x [H = 32] x [W = 32]) and label (N x 1).

    """

    def __init__(self, root, train=True, subset=None, transform=None,
                 target_transform=None):
        """CIFAR-10 dataset constructor

        Parameters
        ----------
        root : str
            Data folder.
        train : bool, optional
            Returns training set if True and test set if False.
            The default is True (training set).
        subset : array, list, or tensor, optional
            Subset of indices of the dataset to use.
            The default is None (use the whole dataset).
        transform : callable, optional
            Transformation to apply to the data points. The default is None.
        target_transform : callable, optional
            Transformation to apply to the labels. The default is None.

        """
        self.classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
                        'Horse', 'Ship', 'Truck')
        self.train = train

        if self.train:
            self.data = torch.load(os.path.join(root, 'cifar10_trainX'))
            self.target = torch.load(os.path.join(root, 'cifar10_trainY'))
        else:
            self.data = torch.load(os.path.join(root, 'cifar10_testX'))
            self.target = torch.load(os.path.join(root, 'cifar10_testY'))

        if subset is not None:
            self.data = self.data[subset,]
            self.target = self.target[subset]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, target = self.data[index,], self.target[index]
        
        # Unsqueeze if single data point
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.target.shape[0]


class FMNIST:
    """FASHION MNIST dataset

    You can download the dataset in PyTorch tensor format from
    https://www.seas.upenn.edu/~luizf/data/fmnist.zip

    ..warning:: For performance purposes, this class loads the full
                FMNIST dataset to RAM. Even though it is less than 1 GB,
                you've been warned


    Attributes
    ----------
    classes : list[str]
        Class labels
    train : bool
        True if training set or False otherwise.
    data : torch.tensor
        FMNIST images.
    transform : callable
        Function applied to the data points before returning them.
    target : torch.tensor
        FMNIST labels.
    target_transform : callable
        Function applied to the labels before returning them.

    Methods
    -------
    __len__()
        Returns size of dataset.
    __get_item__()
        Return tuple (`torch.tensor`, `torch.tensor`) of images
        ([N] x [C = 1] x [H = 28] x [W = 28]) and label (N x 1).

    """

    def __init__(self, root, train=True, subset=None, transform=None,
                 target_transform=None):
        """FASHION MNIST dataset constructor

        Parameters
        ----------
        root : str
            Data folder.
        train : bool, optional
            Returns training set if True and test set if False.
            The default is True (training set).
        subset : array, list, or tensor, optional
            Subset of indices of the dataset to use.
            The default is None (use the whole dataset).
        transform : callable, optional
            Transformation to apply to the data points. The default is None.
        target_transform : callable, optional
            Transformation to apply to the labels. The default is None.

        """
        self.classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        self.train = train

        if self.train:
            self.data = torch.load(os.path.join(root, 'fmnist_trainX'))
            self.target = torch.load(os.path.join(root, 'fmnist_trainY'))
        else:
            self.data = torch.load(os.path.join(root, 'fmnist_testX'))
            self.target = torch.load(os.path.join(root, 'fmnist_testY'))

        if subset is not None:
            self.data = self.data[subset,]
            self.target = self.target[subset]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, target = self.data[index,], self.target[index]
        
        # Unsqueeze if single data point
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.target.shape[0]


class Adult:
    """UCI's adult dataset

    You can download `adult.data` and `adult.test` from
    http://archive.ics.uci.edu/ml/datasets/Adult


    Attributes
    ----------
    classes : list[str]
        Class labels.
    train : bool
        True if training set or False otherwise.
    data : torch.tensor
        FMNIST images.
    transform : callable
        Function applied to the data points before returning them.
    target : torch.tensor
        FMNIST labels.
    target_transform : callable
        Function applied to the labels before returning them.

    Methods
    -------
    __len__()
        Returns size of dataset.
    __get_item__()
        Return tuple (`torch.tensor`, `torch.tensor`) of features (N x F)
        and label (N x 1). The number of features F depends on preprocessing
        (see ``preprocess``).

    """

    categorical = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'gender', 'native-country', 'income']
    """List of categorical variable names (list[str])."""

    def __init__(self, root, target_name='income', train=True, preprocess=None,
                 subset=None, transform=None, target_transform=None):
        """UCI's adult dataset constructor

        Parameters
        ----------
        root : str
            Data folder.
        target_name : str, optional
            Name of target variable. The default is `income`.
        train : bool, optional
            Returns training set if True and test set if False.
            The default is True (training set).
        preprocess : callable, optional
            Transformations to apply before separating labels
            (e.g., binning, dummifying, etc.).
        subset : array, list, or tensor, optional
            Subset of indices of the dataset to use.
            The default is None (use the whole dataset).
        transform : callable, optional
            Transformation to apply to the data points. The default is None.
        target_transform : callable, optional
            Transformation to apply to the labels. The default is None.

        """
        self.classes = ('<= 50k', '> 50k')
        self.train = train

        # Read CSV file
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                        'marital-status', 'occupation', 'relationship', 'race',
                        'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country','income']

        # Load data
        if self.train:
            self.data = pd.read_csv(os.path.join(root, 'adult.data'), sep = ",\s",
                                    header = None, names = column_names, engine = 'python')
        else:
            self.data = pd.read_csv(os.path.join(root, 'adult.test'), sep = ",\s",
                                    header = None, names = column_names, skiprows = 1, engine = 'python')
            self.data['income'].replace(regex = True, inplace = True, to_replace = r'\.', value = r'')

        # Declare categorical variables
        for var_name in Adult.categorical:
            self.data[var_name] = self.data[var_name].astype('category')

        # Preprocess data
        if preprocess is not None:
            self.data = preprocess(self.data)

        # Subset dataset
        if subset is not None:
            if type(subset) is int:
                self.data = self.data.iloc[[subset]]
            else:
                self.data = self.data.iloc[subset]

        # Recompute indices
        self.data.reset_index(drop=True, inplace=True)

        # Recover response variable
        self.target = self.data.filter(regex=f'^{target_name}', axis = 1)
        self.data = self.data.drop(self.target.columns, axis = 1)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if type(index) is int:
            data, target = self.data.iloc[[index]], self.target.iloc[[index]]
        else:
            data, target = self.data.iloc[index], self.target.iloc[index]
        
        # Unsqueeze if single data point
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.target.shape[0]


class COMPAS:
    """ProPublica's COMPAS dataset

    You can download `compas-scores-two-years.csv` from
    https://github.com/propublica/compas-analysis


    Attributes
    ----------
    classes : list[str]
        Class labels.
    train : bool
        True if training set or False otherwise.
    data : torch.tensor
        FMNIST images.
    transform : callable
        Function applied to the data points before returning them.
    target : torch.tensor
        FMNIST labels.
    target_transform : callable
        Function applied to the labels before returning them.

    Methods
    -------
    __len__()
        Returns size of dataset.
    __get_item__()
        Return tuple (`torch.tensor`, `torch.tensor`) of features (N x F)
        and label (N x 1). The number of features F depends on preprocessing
        (see ``preprocess``).

    """

    variables = ['sex', 'age', 'age_cat', 'race', 'decile_score', 'score_text',
                 'v_decile_score', 'v_score_text', 'juv_misd_count', 'juv_other_count',
                 'priors_count', 'c_charge_degree', 'is_recid', 'is_violent_recid',
                 'two_year_recid']
    """List of variables retained from original ProPublica dataset (list[str])."""

    categorical = ['sex', 'age_cat', 'race', 'score_text', 'v_score_text',
                   'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid']
    """List of categorical variable names (list[str])."""

    def __init__(self, root, target_name='two_year_recid', train=True, split=0.7,
                 preprocess=None, subset=None, transform=None, target_transform=None):
        """ProPublica's COMPAS dataset constructor

        Parameters
        ----------
        root : str
            Data folder.
        target_name : str, optional
            Name of target variable. The default is `two_year_recid`.
        train : bool, optional
            Returns training set if True and test set if False.
            The default is True (training set).
        split : float, optional
            Percentage of dataset to keep for training. The dataset is split
            randomly between training and testing, but training and test
            set are deterministic, i.e., the sets returned are always the same.
            The default is 0.7.
        preprocess : callable, optional
            Transformations to apply before separating labels
            (e.g., binning, dummifying, etc.).
        subset : array, list, or tensor, optional
            Subset of indices of the dataset to use.
            The default is None (use the whole dataset).
        transform : callable, optional
            Transformation to apply to the data points. The default is None.
        target_transform : callable, optional
            Transformation to apply to the labels. The default is None.

        """
        self.train = train

        # Read CSV file
        self.data = pd.read_csv(os.path.join(root, 'compas-scores-two-years.csv'))

        # Drop repeated columns
        self.data = self.data.drop('decile_score.1', axis = 1)
        self.data = self.data.drop('priors_count.1', axis = 1)

        # Filter |days_b_screening_arrest| <= 30 (as in ProPublica analysis)
        self.data = self.data[(self.data['days_b_screening_arrest'] >= -30) &
                              (self.data['days_b_screening_arrest'] <= 30)]

        # Random split
        N = self.data.shape[0]
        idx_list = np.random.RandomState(seed=42).permutation(N)
        split_idx = int(np.ceil(N*split))
        train_idx = idx_list[:split_idx]
        test_idx = idx_list[split_idx:]

        # Normalize indices
        self.data.reset_index(drop=True, inplace=True)

        if self.train:
            self.data = self.data.iloc[train_idx,]
        else:
            self.data = self.data.iloc[test_idx,]

        # Renomarlize indices
        self.data.reset_index(drop=True, inplace=True)

        # Keep only columns of interest
        self.data = self.data[COMPAS.variables]

        # Declare categorical variables
        for var_name in COMPAS.categorical:
            self.data[var_name] = self.data[var_name].astype('category')

        if preprocess is not None:
            self.data = preprocess(self.data)

        # Subset data
        if subset is not None:
            if type(subset) is int:
                self.data = self.data.iloc[[subset]]
            else:
                self.data = self.data.iloc[subset]

        # Recompute indices
        self.data.reset_index(drop=True, inplace=True)

        # Recover response variable
        self.target = self.data.filter(regex=f'^{target_name}', axis = 1)
        self.data = self.data.drop(self.target.columns, axis = 1)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if type(index) is int:
            data, target = self.data.iloc[[index]], self.target.iloc[[index]]
        else:
            data, target = self.data.iloc[index], self.target.iloc[index]

        # Unsqueeze if single data point
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.target.shape[0]


class UTK:
    """UTKFace dataset

    Download the dataset from https://susanqq.github.io/UTKFace/ and indicate
    the path to the UTKFace folder 


    Attributes
    ----------
    classes : list[str]
        Class labels
    train : bool
        True if training set or False otherwise.
    current_batch : dict
        Memoized dataset to speed-up consecutive requests for the same data.
    data : panda
        Panda data frame containing the targets and path to each image.
        Contrary to `CIFAR-10` or `FMNIST`, UTKFace is never fully loaded
        into memory.
    transform : callable
        Function applied to the data points before returning them.
    target_transform : callable
        Function applied to the labels before returning them.

    Methods
    -------
    __len__()
        Return size of dataset.
    __get_item__()
        Return tuple (`torch.tensor`, `pandas`) of image
        ([N] x [C = 3] x [H = 200] x [W = 200]) and label (N x 3).

    """

    def __init__(self, root, train=True, split=0.7, preprocess=None,
                 subset=None, transform=None, target_transform=None):
        """UTKFace dataset constructor

        Parameters
        ----------
        root : str
            Data folder.
        train : bool, optional
            Returns training set if True and test set if False.
            The default is True (training set).
        split : float, optional
            Percentage of dataset to keep for training. The dataset is split
            randomly between training and testing, but training and test
            set are deterministic, i.e., the sets returned are always the same.
            The default is 0.7.
        preprocess : callable, optional
            Transformations to apply before separating labels
            (e.g., binning, dummifying, etc.).
        subset : array, list, or tensor, optional
            Subset of indices of the dataset to use.
            The default is None (use the whole dataset).
        transform : callable, optional
            Transformation to apply to the data points. The default is None.
        target_transform : callable, optional
            Transformation to apply to the labels. The default is None.

        """
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.current_batch = {'batch_idx': None,
                              'data': None,
                              'target': None}

        # Load dataset
        files = glob.glob(os.path.join(root, 'UTKFace', '*.jpg'))
        self.data = [self._parse_file(file) for file in files]
        self.data = pd.DataFrame(self.data)
        self.data.columns = ['age', 'gender', 'race']
        self.data['filename'] = files

        # Keep complete cases
        self.data = self.data.dropna()

        # Renomarlize indices
        self.data.reset_index(drop=True, inplace=True)

        # Set categorical variables
        # {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
        self.data['race'] = self.data['race'].astype('category')
        self.data['gender'] = self.data['gender'].astype('category')

        # Random split
        N = self.data.shape[0]
        idx_list = np.random.RandomState(seed=42).permutation(N)
        split_idx = int(np.ceil(N*split))
        train_idx = idx_list[:split_idx]
        test_idx = idx_list[split_idx:]

        if self.train:
            self.data = self.data.iloc[train_idx,]
        else:
            self.data = self.data.iloc[test_idx,]

        # Renomarlize indices
        self.data.reset_index(drop=True, inplace=True)

        # Preprocess data
        if preprocess is not None:
            self.data = preprocess(self.data)

        # Subset data
        if subset is not None:
            if type(subset) is int:
                self.data = self.data.iloc[[subset]]
            else:
                self.data = self.data.iloc[subset]

        # Renomarlize indices
        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        # Get data subset
        if type(index) is int:
            df = self.data.iloc[[index]]
        else:
            df = self.data.iloc[index]

        if self.current_batch['batch_idx'] == set(df.index.values):
            # Load memoized batch
            samples = self.current_batch['data']
            target = self.current_batch['target']
        else:
            # Load batch from memory
            samples = [self._image_to_tensor(filename) for filename in df['filename']]
            samples = torch.stack(samples, dim=0).squeeze()
            if len(df) == 1:
                samples = samples.unsqueeze(0)
            target = df[['age', 'gender', 'race']]
            
            if self.transform is not None:
                samples = self.transform(samples)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # Memoize batch
            self.current_batch['batch_idx'] = set(df.index.values)
            self.current_batch['data'] = samples
            self.current_batch['target'] = target

        return samples, target

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def _parse_file(filename):
        """
        Extract information about data point from filename.
        """
        try:
            age, gender, race, _ = os.path.split(filename)[1].split('_')
            return int(age), int(gender), int(race)
        except Exception:
            return None, None, None

    @staticmethod
    def _image_to_tensor(filename):
        """
        Convert PIL image to tensor and normalize values to [0,1]
        """
        # Load image
        pic = Image.open(filename)
        pic = pic.resize((100,100))

        # Convert to tensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))

        # Convert to C x H x W format
        img = img.permute((2, 0, 1)).contiguous()

        # Return [0,1] image
        return img.float()/255.0
