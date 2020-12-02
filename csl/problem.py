# -*- coding: utf-8 -*-
"""Constrained learning problem base class

In csl, constrained learning problems are specified by inheriting from
`ConstrainedLearningProblem` and defining its attributes.

"""

import torch
import numpy as np
from csl.utils import _batches


class ConstrainedLearningProblem:
    """Constrained learning problem base class.

    Constrained learning problems are defined by inheriting from
    `ConstrainedLearningProblem` and defining its attributes:

    - ``model``: underlying model to train
    - ``data``: data with which to train the model
    - ``batch_size`` (optional): maximum number of data points to load to memory at once
    - ``obj_function``: objective function or training loss
    - ``constraints`` (optional): average constraints
    - ``rhs`` (optional): right-hand side of average constraints
    - ``pointwise`` (optional): pointwise constraints
    - ``pointwise_rhs`` (optional): right-hand side of pointwise constraints

    A detailed description of each of these attributes is given below.

    Attributes
    ----------
    model : `callable`
        Model used to solve the constrained learning problem. The model must have
        an attribute ``parameters`` and a method ``__call__`` as specified below.

        - ``parameters``: model parameters (`list` [`torch.tensor`] with ``requires_grad=True``)
        - ``__call__(x)``: takes a data batch ``x`` and evaluates the output of the
          model for each data point in ``x`` (`callable`)

    data : `list`
        Training data. Must define the methods ``__len__`` and ``__get_item__``:

        - ``__len__``: Returns size of dataset (callable)
        - ``__get_item__``: Returns element(s) from dataset (callable)

    batch_size : `int`
        Internal batch size to evaluate empirical averages.

        ..Note:: this has no effect on the training batch size. It is only
                 used internally to avoid running out of memory when evaluating
                 quantities that require a full pass over the dataset.

    obj_function : `callable`
        Objective function. Takes a list of indices (`list` [`int`]) defining a
        mini-batch and returns the objective function value (`torch.tensor`, (1, ))
        over that mini-batch.

    constraints : `list` [`callable`]
        Functions defining the average constraints. Each function takes

        - ``batch_idx``: list of indices defining a mini-batch (`list` [`int`])
        - ``primal``: `True` if constraint is being evaluated for primal update
          or `False` otherwise (`bool`)

        and returns the average constraint value (`torch.tensor`, (1, ))
        over that mini-batch.

    rhs : `list` [`float`]
        List containing the right-hand side of each average constraint.

    pointwise : `list` [`callable`]
        Functions defining the pointwise constraints. Each function takes

        - ``batch_idx``: list of indices defining a mini-batch (`list` [`int`])
        - ``primal``: `True` if constraint is being evaluated for primal update
          or `False` otherwise (`bool`)

        and returns the pointwise constraint value (`torch.tensor`, (``len(batch_idx)``, ))
        for each point in the mini-batch.

    pointwise_rhs : `list` [`torch.tensor`, (N, )]
        List containing the right-hand side of each pointwise constraint.


    Notes
    -----

    **The primal flag**

    When working with non-differentiable constraints, a smooth approximation
    can be used during the primal computation to enable gradient updates. If this
    approximation is good enough, i.e., if the minimum of the smooth function is a
    good approximation of the minimum of the non-differentiable function, then
    certain guarantees can be given on the solutions obtained by the primal-dual
    iterations.

    The purpose of this flag is to allow for these alternative smooth approximations
    to be used when minimizing the Lagrangian using gradient descent. The original,
    non-differentiable loss is then used during the dual update to compute the
    supergradients of the Lagrangian with respect to the dual variables.


    **An example problem**

    To pose a constrained learning problem, inherit from `ConstrainedLearningProblem`
    and define its attributes before initializing the base class by calling
    ``super().__init__()``.

     .. code-block:: python
        :linenos:

        import torch
        import torch.nn.functional as F
        import csl

        ####################################
        # MODEL                            #
        ####################################
        class Logistic:
            def __init__(self, n_features):
                self.parameters = [torch.zeros(1, dtype = torch.float, requires_grad = True),
                                   torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

            def __call__(self, x):
                if len(x.shape) == 1:
                    x = x.unsqueeze(1)

                yhat = self.logit(torch.mm(x, self.parameters[1]) + self.parameters[0])

                return torch.cat((1-yhat, yhat), dim=1)

            def predict(self, x):
                _, predicted = torch.max(self(x), 1)
                return predicted

            @staticmethod
            def logit(x):
                return 1/(1 + torch.exp(-x))

        ####################################
        # PROBLEM                          #
        ####################################
        class fairClassification(csl.ConstrainedLearningProblem):
            def __init__(self):
                self.model = Logistic(data[0][0].shape[0])
                self.data = data
                self.obj_function = self.loss

                # Demographic parity
                self.constraints = [ self.demographic_parity ]
                self.rhs = [ 0.1 ]

                super().__init__()

            def loss(self, batch_idx):
                # Evaluate objective
                x, y = self.data[batch_idx]
                yhat = self.model(x)

                return F.cross_entropy(yhat, y)

            def demographic_parity(self, batch_idx, primal):
                    protected_idx = 3
                    x, y = self.data[batch_idx]
                    group_idx = (x[:,protected_idx] == 1)

                    if primal:
                        # Sigmoid approximation of indicator function
                        yhat = self.model(x)
                        pop_indicator = torch.sigmoid(8*(yhat[:,1] - 0.5))
                        group_indicator = torch.sigmoid(8*(yhat[group_idx,1] - 0.5))
                    else:
                        # Indicator function
                        yhat = self.model.predict(x)
                        pop_indicator = yhat.float()
                        group_indicator = yhat[group_idx].float()

                    return pop_indicator.mean() - group_indicator.mean()


    """

    def __init__(self):
        # Check subclassing definition
        if not hasattr(self, 'model'):
            raise Exception('Your CSL problem must have a model.')

        if not hasattr(self, 'data'):
            raise Exception('Your CSL problem must have data.')

        if not hasattr(self, 'obj_function'):
            raise Exception('Your CSL problem must have an objective function.')

        # Finish initializing problem
        model_device = next(iter(self.model.parameters)).device

        if not hasattr(self, 'batch_size'):
            self.batch_size = None

        if not hasattr(self, 'data_size'):
            self.data_size = len(self.data)

        if not hasattr(self, 'constraints'):
            # Takes batch indices, returns a scalar average value
            self.constraints = []
            self.rhs = []
            self.lambdas = []
        else:
            self.lambdas = [torch.tensor(0, dtype = torch.float,
                                         requires_grad = False,
                                         device = model_device) \
                            for _ in self.constraints]

        if not hasattr(self, 'pointwise'):
            # Takes batch indices, returns a vector with one element per data point
            self.pointwise = []
            self.pointwise_rhs = []
            self.mus = []
        else:
            self.mus = [torch.zeros_like(rhs, dtype = torch.float,
                                         requires_grad = False,
                                         device = model_device) \
                        for rhs in self.pointwise_rhs]


    def lagrangian(self, batch_idx=None):
        """Evaluate Lagrangian (and its gradient)

        Parameters
        ----------
        batch_idx : `list` [`int`], optional
            Indices of batch. The default is `None` to evaluate over the full dataset.
            The evaluation is done in batches size according to ``batch_size`` to
            avoid loading the full dataset to the memory.

        Returns
        -------
        L : `float`
            Lagrangian value.
        obj_value : `float`
            Objective value.
        constraints_slacks : `list` [`torch.tensor`, (1, )]
            Slacks of average constraints
        pointwise_slacks : `list` [`torch.tensor`, (``len(batch_idx)``, )]
            Slacks of pointwise constraints
        """
        if batch_idx is not None:
            L, obj_value, constraint_slacks, pointwise_slacks = self._lagrangian(batch_idx)
        else:
            # Initialization
            L = 0
            obj_value = 0
            constraint_slacks = [0]*len(self.constraints)
            pointwise_slacks = [torch.zeros([0])]*len(self.pointwise)

            # Compute over the whole data set in batches
            for batch_start, batch_end in _batches(self.data_size, self.batch_size):
                L_batch, obj_value_batch, constraint_slacks_batch, pointwise_slacks_batch = self._lagrangian(np.arange(batch_start,batch_end))

                L += L_batch*(batch_end - batch_start)/self.data_size

                obj_value += obj_value_batch*(batch_end - batch_start)/self.data_size

                for ii, slack in enumerate(constraint_slacks_batch):
                    constraint_slacks[ii] += slack*(batch_end - batch_start)/self.data_size

                for ii, slack in enumerate(pointwise_slacks_batch):
                    pointwise_slacks[ii] = torch.cat((pointwise_slacks[ii], slack))

        return L, obj_value, constraint_slacks, pointwise_slacks


    def objective(self, batch_idx=None):
        """Evaluate the objective function

        Parameters
        ----------
        batch_idx : `list` [`int`], optional
            Indices of batch. The default is `None` to evaluate over the full dataset.
            The evaluation is done in batches size according to ``batch_size`` to
            avoid loading the full dataset to the memory.

        Returns
        -------
        obj_value : `float`
            Objective value.

        """
        if batch_idx is not None:
            obj_value = self.obj_function(batch_idx).item()
        else:
            obj_value = 0
            for batch_start, batch_end in _batches(self.data_size, self.batch_size):
                obj_value += self.obj_function(range(batch_start,batch_end)).item()*(batch_end - batch_start)/self.data_size

        return obj_value


    def slacks(self, batch_idx=None):
        """Evaluate constraint slacks

        Parameters
        ----------
        batch_idx : `list` [`int`], optional
            Indices of batch. The default is `None` to evaluate over the full dataset.
            The evaluation is done in batches size according to ``batch_size`` to
            avoid loading the full dataset to the memory.

        Returns
        -------
        constraint_slacks : `list` [`float`]
            Constraint violation of the average constraints.
        pointwise_slacks : `list` [`torch.tensor`, (``len(batch_idx)``, )]
            Constraint violation of the pointwise constraints.

        """
        if batch_idx is not None:
            constraint_slacks = self._constraint_slacks(batch_idx)
            pointwise_slacks = self._pointwise_slacks(batch_idx)
        else:
            constraint_slacks = [0]*len(self.constraints)
            pointwise_slacks = [torch.zeros([0])]*len(self.pointwise)

            for batch_start, batch_end in _batches(self.data_size, self.batch_size):
                for ii, s in enumerate(self._constraint_slacks(range(batch_start,batch_end))):
                    constraint_slacks[ii] += s*(batch_end - batch_start)/self.data_size
                for ii, s in enumerate(self._pointwise_slacks(range(batch_start,batch_end))):
                    pointwise_slacks[ii] = torch.cat((pointwise_slacks[ii], s))

        return constraint_slacks, pointwise_slacks



    ###########################################################################
    #### PRIVATE FUNCTIONS                                                 ####
    ###########################################################################
    def _constraint_slacks(self, batch_idx):
        """Evaluate constraint slacks for average constraints over batch

        Parameters
        ----------
        batch_idx : `list` [`int`]
            Indices of batch.

        Returns
        -------
        slacks_value : `list` [`float`]
            Constraint violation of the average constraints.

        """
        slacks_value = [ell(batch_idx, primal=False) - c for ell, c in zip(self.constraints, self.rhs)]
        return slacks_value


    def _pointwise_slacks(self, batch_idx):
        """Evaluate constraint slacks for pointwise constraints over batch

        Parameters
        ----------
        batch_idx : `list` [`int`]
            Indices of batch.

        Returns
        -------
        slacks_value : `list` [`torch.tensor`, (``len(batch_idx)``, )]
            Constraint violation of the pointwise constraints.

        """
        slacks_value = [ell(batch_idx, primal=False) - c[batch_idx] for ell, c in zip(self.pointwise, self.pointwise_rhs)]
        return slacks_value


    def _lagrangian(self, batch_idx):
        """Evaluate Lagrangian over batch

        Parameters
        ----------
        batch_idx : `list` [`int`]
            Indices of batch.

        Returns
        -------
        L : `float`
            Lagrangian value.
        obj_value : `float`
            Objective value.
        constraints_slacks : `list` [`torch.tensor`, (1, )]
            Slacks of average constraints
        pointwise_slacks : `list` [`torch.tensor`, (``len(batch_idx)``, )]
            Slacks of pointwise constraints
        """
        L = 0
        constraints_slacks = []
        pointwise_slacks = []

        # Objective value
        obj_value = self.obj_function(batch_idx)
        if torch.is_grad_enabled():
            obj_value.backward()
        L += obj_value.item()

        # Dualized average constraints
        for lambda_value, ell, c in zip(self.lambdas, self.constraints, self.rhs):
            slack = ell(batch_idx, primal=True) - c
            dualized_slack = lambda_value*slack
            if torch.is_grad_enabled():
                dualized_slack.backward()

            constraints_slacks += [slack]
            L += dualized_slack.item()

        # Dualized pointwise constraints
        for mu_value, ell, c in zip(self.mus, self.pointwise, self.pointwise_rhs):
            slack = ell(batch_idx, primal=True) - c[batch_idx]
            dualized_slack = torch.dot(mu_value[batch_idx], slack)/len(batch_idx)
            if torch.is_grad_enabled():
                dualized_slack.backward()

            pointwise_slacks += [slack]
            L += dualized_slack.item()

        return L, obj_value.item(), constraints_slacks, pointwise_slacks
