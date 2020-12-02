Getting started
===============

Installation
------------

In your working folder simply do
::

    git clone https://github.com/lchamon/csl.git


or `download <https://github.com/lchamon/csl/archive/main.zip>`_ and extract.

You will need to have

* numpy
* pytorch
* matplotlib (for plotting)


A dummy dataset
---------------

In the following examples, we consider some noisy data generated using a linear model.

.. code-block:: python
    :linenos:

    class linearData:
        def __init__(self, dim, n):
            self.wo = torch.ones(dim,1)
            self.x = torch.randn(n,dim)
            self.y = torch.mm(self.x, self.wo) + torch.sqrt(1e-3)*torch.randn(n,1)

        def __getitem__(self, idx):
            return self.x[idx,:], self.y[idx]

        def __len__(self):
            return self.x.shape[0]


A **csl** model
---------------

You can use any pytorch model you want with **csl**. However, it must have at
least an attribute ``parameters`` and a method ``__call__``

- ``parameters``: model parameters (`list` [`torch.tensor`])
- ``__call__(x)``: takes a data batch ``x`` and evaluates the output of the
  model for each data point in ``x`` (`callable`)

Unless you write your own solver which uses a different way to optimize the
model parameters, they should be a list of `torch.tensor` with ``requires_grad=True``.

For instance, let's consider the linear model (without intercept):

.. code-block:: python
    :linenos:

    class Linear:
        def __init__(self, n_features):
            self.parameters = [torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

        def __call__(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(1)

            yhat = torch.mm(x, self.parameters[0])

            return yhat.squeeze()

        def predict(self, x):
            return self(x)

Since this is not exactly the interface you get for a pytorch neural network,
**csl** provides the wrapper :py:mod:`csl.models.PytorchModel` you can use
around your favorite pytorch model by simply doing ``csl.PytorchModel(resnet.ResNet18())``.


Defining a problem
------------------
To define a constrained learning problem, inherit from
:py:mod:`csl.problem.ConstrainedLearningProblem` and define its attributes.
You must provide at least

* ``model``: model to train
* ``data``: data with which to train the model
* ``obj_function``: objective function or training loss

Additionally, if your dataset is too large to fit in memory, you may want to include

* ``batch_size`` (optional): maximum number of points to load to memory at once

This is only used to evaluate internal problem quantities and is completely
independent from the solver mini-batch size (see `Setting up the solver`_).

At this point, you have an unconstrained (classical) learning problem. If you throw it
at a **csl** solver, it will be exactly as if you were using vanilla pytorch.
So you might want to also include constraints using

* ``constraints`` (optional): average constraints
* ``rhs`` (optional): right-hand side of average constraints
* ``pointwise`` (optional): pointwise constraints
* ``pointwise_rhs`` (optional): right-hand side of pointwise constraints

..note:: After defining these attributes, do not forget to call the base
         class constructor using ``super().__init__()``.

A **csl** problem might look like this:

.. code-block:: python
    :linenos:

    class QCQP(csl.ConstrainedLearningProblem):
        def __init__(self):
            self.model = Linear(10)         # Insert your model here
            self.data = linearData(10,100)  # Insert your dataset here

            # Objective function
            self.obj_function = self.loss

            # Average constraints
            self.constraints = [lambda batch, primal: torch.mean(self.model.parameters[0]**2)]
            self.rhs = [0.5]

            # Pointwise constraints
            self.pointwise = [self.pointwise_loss]
            self.pointwise_rhs = [5*torch.ones(len(data), requires_grad = False)]

            super().__init__()

        def loss(self, batch_idx):
            # Get data batch
            x, y = self.data[batch_idx]

            # Compute model output
            yhat = self.model(x)

            # Return average loss
            return torch.mean((yhat - y.squeeze())**2)

        def pointwise_loss(self, batch_idx, primal):
            # Get data batch
            x, y = self.data[batch_idx]

            # Compute model output
            yhat = self.model(x)

            # Return square loss for each data point
            return (yhat - y.squeeze())**2


After that, you still need to build yourself a problem using ``problem = QCQP()``.
You can also include variables in the constructor to make your problem parametric.
For instance, you could want to solve ``QCQP`` for different specifications of
the constraints.



Setting up the solver
---------------------

Now that we have data, model, and problem, the only thing we are missing is a solver.
Right now, **csl** has two primal-dual solvers: :py:mod:`csl.solvers.PrimalThenDual`
(or just ``PrimalDual`` for short) or :py:mod:`csl.solvers.SimultaneousPrimalDual`.
They differ only the scheduling between the primal and dual updates.
Essentially, :py:mod:`csl.solvers.PrimalThenDual` updates the dual variables at the end
of each epoch, whereas :py:mod:`csl.solvers.SimultaneousPrimalDual`
updates the dual variables for every mini-batch.

For all intents and purposes, you could just take the default settings and go
with ``solver = csl.PrimalDual()``. They are not great default settings though.
So you might want to set up your problem a bit as in

.. code-block:: python
    :linenos:

    solver_settings = {'iterations': 2000,
                       'batch_size': 10,
                       'lr_p0': 0.01,
                       'lr_d0': 0.01,
                       }

    solver = csl.PrimalDual(solver_settings)


You can find a complete list of settings and defaults at :py:mod:`csl.solver_base.SolverSettings`
and in the description of the specific solvers (:py:mod:`csl.solvers`).


Putting it all together
-----------------------

With your solver and problem in hand, all you need to do is ``solver.solve(problem)``.
You can see trace plots once the solver finishes using ``solver.plot()``. You can reuse
the same solver for other problems (or the same problem with other parameters) by first
calling ``solver.reset()``.

.. code-block:: python
    :linenos:

    import torch
    import csl

    torch.manual_seed(1234)

    ####################################
    # SIMULATED DATA                   #
    ####################################
    class linearData:
        def __init__(self, dim, n):
            self.wo = torch.ones(dim,1)
            self.x = torch.randn(n,dim)
            self.y = torch.mm(self.x, self.wo) + torch.sqrt(1e-3)*torch.randn(n,1)

        def __getitem__(self, idx):
            return self.x[idx,:], self.y[idx]

        def __len__(self):
            return self.x.shape[0]

    ####################################
    # LINEAR MODEL                     #
    ####################################
    class Linear:
        def __init__(self, n_features):
            self.parameters = [torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

        def __call__(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(1)

            yhat = torch.mm(x, self.parameters[0])

            return yhat.squeeze()

        def predict(self, x):
            return self(x)

    ####################################
    # CSL PROBLEM                      #
    ####################################
    class QCQP(csl.ConstrainedLearningProblem):
        def __init__(self):
            self.model = Linear(10)
            self.data = linearData(10,100)

            self.obj_function = self.loss
            self.constraints = [lambda batch, primal: torch.mean(self.model.parameters[0]**2)]
            self.rhs = [0.5]
            self.pointwise = [self.pointwise_loss]
            self.pointwise_rhs = [5*torch.ones(len(data), requires_grad = False)]

            super().__init__()

        def loss(self, batch_idx):
            # Evaluate objective
            x, y = self.data[batch_idx]
            yhat = self.model(x)

            return torch.mean((yhat - y.squeeze())**2)
            # return torch.ones(1, requires_grad=True)

        def pointwise_loss(self, batch_idx, primal):
            # Evaluate objective
            x, y = self.data[batch_idx]
            yhat = self.model(x)

            return (yhat - y.squeeze())**2

    problem = QCQP()

    ####################################
    # CSL SOLVER                       #
    ####################################
    solver_settings = {'iterations': 2000,
                       'batch_size': 10,
                       'lr_p0': 0.01,
                       'lr_d0': 0.01,
                       }

    solver = csl.PrimalDual(solver_settings)

    ####################################
    # TRAINING                         #
    ####################################
    solver.solve(problem)
    solver.plot()
