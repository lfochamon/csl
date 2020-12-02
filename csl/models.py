# -*- coding: utf-8 -*-
"""Model wrappers for csl module

"""

import torch

class PytorchModel:
    """PyTorch model wrapper for constrained learning problems.

    Attributes
    ----------
    model : `torch.nn.Module`
        A PyTorch model.

    parameters : `list` [`torch.tensor`]
        Model parameters. Obtained directly from the underlying PyTorch module as
        ``model.parameters()``. Setting ``parameters``, however, expects a module
        state dictionary obtained by calling ``model.state_dict``.

    """

    def __init__(self, model):
        self.__dict__['model'] = model

    def __call__(self, x):
        """Evaluate model output

        Parameters
        ----------
        x : `torch.tensor`
            Input data
        """
        return self.model(x)

    def predict(self, x):
        """Evaluate model prediction

        Predicts the label of each data point in ``x``.
        Assumes the neural network has one output per class and
        returns the class corresponding to the largest output.

        Parameters
        ----------
        x : `torch.tensor`
            Input data
        """
        _, predicted = torch.max(self(x), 1)
        return predicted

    def __getattr__(self, attr):
        """Get model attribute

        Passed directly to the underlying PyTorch model except for ``parameters``
        which returns a generator over the model parameters.

        Parameters
        ----------
        attr : `str`
            Attribute name
        """
        if attr == 'parameters':
            return self.__dict__['model'].parameters()
        else:
            return getattr(self.model, attr)

    def __setattr__(self, attr, value):
        """Set model attribute

        Passed directly to the underlying PyTorch model except for ``parameters``
        which expects a complete model state dictionary

        Parameters
        ----------
        attr : `str`
            Attribute name
        """
        if attr == 'parameters':
            self.model.load_state_dict(value)
        else:
            setattr(self.model, attr, value)
