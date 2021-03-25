# -*- coding: utf-8 -*-
"""Dataset transformations

"""

import pandas as pd
import torch


class Drop:
    """Remove variables from data frame.

    Attributes
    ----------
    var_name : `list` [`str`]
        Variable names.

    """
    def __init__(self, var_names):
        self.var_names = var_names

    def __call__(self, sample):
        """Remove variables from data frame.

        Parameters
        ----------
        sample : `pandas.DataFrame`
            Data frame.

        Returns
        -------
        `pandas.DataFrame`
            Data frame without variables.

        """
        return sample.drop(self.var_names, axis = 1)


class Recode:
    """Recode variable.

    Attributes
    ----------
    var_name : `str`
        Variable name.
    dictionary : `dict`
        Dictionary describing recoding patterns, e.g.,
        ``{'L': ['L1', 'L2']}`` recodes levels ``L1`` and ``L2`` as ``L``

    """

    def __init__(self, var_name, dictionary):
        self.var_name = var_name
        self.dictionary = dictionary

    def __call__(self, sample):
        """Recode variable.

        Parameters
        ----------
        sample : `pandas.DataFrame`
            Data frame.

        Returns
        -------
        `pandas.DataFrame`
            Data frame with recoded variable.

        """
        transposed_dicitionary = {}
        for new_value, old_values in self.dictionary.items():
            for value in old_values:
                transposed_dicitionary[value] = new_value

        if isinstance(sample[self.var_name].dtype, pd.CategoricalDtype):
            sample[self.var_name] = sample[self.var_name].replace(transposed_dicitionary).astype('category')
        else:
            sample[self.var_name] = sample[self.var_name].replace(transposed_dicitionary)

        return sample


class Dummify:
    """Dummy code variables.

    Attributes
    ----------
    var_names : `list` [`str`]
        Variable names.

    """

    def __init__(self, var_names):
        self.var_names = var_names

    def __call__(self, sample):
        """Dummy code variables.

        Parameters
        ----------
        sample : `pandas.DataFrame`
            Data frame.

        Returns
        -------
        `pandas.DataFrame`
            Data frame with encoded variables.

        """
        for name in self.var_names:
            if name in sample.columns:
                if len(sample[name].cat.categories) > 2:
                    sample = pd.get_dummies(sample, prefix=[name], columns=[name])
                else:
                    sample = pd.get_dummies(sample, prefix=[name], columns=[name], drop_first=True)
        return sample


class QuantileBinning:
    """Bin variable in quantiles.

    Attributes
    ----------
    var_name : `str`
        Variable names.
    quantile : `int`
        Number of bins.

    """

    def __init__(self, var_name, quantile):
        self.var_name = var_name
        self.quantile = quantile

    def __call__(self, sample):
        """Bin variable in quantiles.

        Parameters
        ----------
        sample : `pandas.DataFrame`
            Data frame.

        Returns
        -------
        `pandas.DataFrame`
            Data frame after binning.

        """
        sample[self.var_name] = pd.qcut(sample[self.var_name], q = self.quantile)

        return sample


class Binning:
    """Bin variable.

    Attributes
    ----------
    var_name : `str`
        Variable name.
    bins : `list` [`int`]
        Bin edges (each bin includes right edge and first bin includes both edges).

    """

    def __init__(self, var_name, bins):
        self.var_name = var_name
        self.bins = bins

    def __call__(self, sample):
        """Bin variable.

        Parameters
        ----------
        sample : `pandas.DataFrame`
            Data frame.

        Returns
        -------
        `pandas.DataFrame`
            Data frame with modified variable.

        """
        sample[self.var_name] = pd.cut(sample[self.var_name], bins = self.bins,
                                       include_lowest = True)

        return sample


class ToTensor:
    """Transform input to `torch.tensor` or cast `torch.tensor` to ``dtype`` and ``device``.

    Attributes
    ----------
    **kwargs : `dict`
        Parameters to pass to tensor constructor.

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sample):
        """Transform input to `torch.tensor` or cast `torch.tensor` to ``dtype`` and ``device``.

        Parameters
        ----------
        sample : `list` or `torch.tensor`
            Object to be cast as a `torch.tensor` or `torch.tensor`.

        Returns
        -------
        `torch.tensor`

        """
        if type(sample) is torch.Tensor:
            return sample.to(dtype = self.kwargs.get('dtype'),
                             device = self.kwargs.get('device'))
        else:
            if type(sample) is pd.DataFrame:
                return torch.tensor(sample.to_numpy(dtype='float'), **self.kwargs).squeeze()
            else:
                # Unknown object, try your best
                return torch.tensor(sample, **self.kwargs).squeeze()


class RandomFlip:
    """Randomly flip image along an axis.

    Attributes
    ----------
    p : `float`, optional
        Flipping probability. The default is 0.5.
    axis : `int`, optional
        Axis along which to flip. The default is 3 (horizontal flip).

    """

    def __init__(self, p = 0.5, axis = 3):
        self.p = p
        self.axis = axis

    def __call__(self, img):
        """Randomly flip image along an axis.

        Parameters
        ----------
        img : `torch.tensor`
            Image batch (N x H x W x C).

        Returns
        -------
        `torch.tensor`

        """
        img = img.clone()
        flipped = torch.rand(img.size(0)) < self.p
        img[flipped] = torch.flip(img[flipped], [3])
        return img


class RandomCrop:
    """Pad and randomly crop image.

    Attributes
    ----------
    size : `int`
        Size of region to crop (in pixels).
    padding : `int`
        Size of padding to add before cropping (in pixels).

    """

    def __init__(self, size, padding):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        """Pad and randomly crop image.

        Parameters
        ----------
        img : `torch.tensor`
            Image batch (N x H x W x C).

        Returns
        -------
        `torch.tensor`

        """
        if self.padding is not None:
            padded = torch.zeros((img.size(0), img.size(1), img.size(2) + self.padding * 2,
                                  img.size(3) + self.padding * 2), dtype=torch.float)
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = img
        else:
            padded = img

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (img.size(0),))
            j = torch.randint(0, w - tw + 1, (img.size(0),))

        rows = torch.arange(th, dtype=torch.long) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(img.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]]

        return padded.permute(1, 0, 2, 3)
