# -*- coding: utf-8 -*-
"""Helper functions for the csl module

"""

import numpy as np

def _batches(length, batch_size):
    """Evaluates batch edges

    Generator for batch edges taking into account possible smaller tail batch.
    Each batch is of the form ``data[start:end]``.

    Parameters
    ----------
    length : `int`
        Length of full dataset.
    batch_size : `int`
        Size of batch.

    Yields
    ------
    start : `int`
        First index of current batch.
    end : `int`
        Last index of current batch.

    """
    if batch_size is None:
        batch_idx = [0, length]
    else:
        batch_idx = np.arange(0, length+1, batch_size)
        if batch_idx[-1] < length:
            batch_idx = np.append(batch_idx, length)

    for start, end in zip(batch_idx, batch_idx[1:]):
        yield start, end
