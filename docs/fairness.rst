We will use the Adult dataset from UCI to demonstrate how to impose fairness
constraints. Here, the goal is to predict whether to grant a loan to an individual
by trying to predict if they make more than US$ 50k per year. However, we want to
make sure that loans are granted as likely to be granted to women than to men.

You can find more information in
`[CR, NeurIPS'20] <https://www.seas.upenn.edu/~luizf/pages/publications.html#Chamon20p>`_.

For this example, you will need to go get ``adult.data`` and ``adult.test`` from
`UCI <http://archive.ics.uci.edu/ml/datasets/Adult>`_ and place them in a folder
named ``data``.

You can try the full code on `GitHub <https://github.com/lchamon/csl>`_.


Basic setup
^^^^^^^^^^^

.. code-block:: python
    :linenos:

    import torch
    import torch.nn.functional as F
    import torchvision

    import matplotlib.pyplot as plot

    import functools

    import sys, os
    sys.path.append(os.path.abspath('../'))

    import csl, csl.datasets


Loading data
^^^^^^^^^^^^

We use :py:mod:`csl.datasets.utils` to do a bit of data wrangling. Drop some variables,
bin others, and dummy code categorical variables.

.. code-block:: python
    :linenos:

    # Preprocessing
    preprocess = torchvision.transforms.Compose([
        csl.datasets.utils.Drop(['fnlwgt', 'educational-num', 'relationship', 'capital-gain', 'capital-loss']),
        csl.datasets.utils.Recode('education', {'<= K-12': ['Preschool', '1st-4th', '5th-6th', '7th-8th',
                                          '9th', '10th', '11th', '12th']}),
        csl.datasets.utils.Recode('race', {'Other': ['Other', 'Amer-Indian-Eskimo']}),
        csl.datasets.utils.Recode('marital-status', {'Married': ['Married-civ-spouse', 'Married-AF-spouse',
                                              'Married-spouse-absent'],
                                  'Divorced/separated': ['Divorced', 'Separated']}),
        csl.datasets.utils.Recode('native-country', {'South/Central America': ['Columbia', 'Cuba', 'Guatemala',
                                                            'Haiti', 'Ecuador', 'El-Salvador',
                                                            'Dominican-Republic', 'Honduras',
                                                            'Jamaica', 'Nicaragua', 'Peru',
                                                            'Trinadad&Tobago'],
                                  'Europe': ['England', 'France', 'Germany', 'Greece',
                                              'Holand-Netherlands', 'Hungary', 'Italy',
                                              'Ireland', 'Portugal', 'Scotland', 'Poland',
                                              'Yugoslavia'],
                                  'Southeast Asia': ['Cambodia', 'Laos', 'Philippines',
                                                      'Thailand', 'Vietnam'],
                                  'Chinas': ['China', 'Hong', 'Taiwan'],
                                  'USA': ['United-States', 'Outlying-US(Guam-USVI-etc)',
                                          'Puerto-Rico']}),
        csl.datasets.utils.QuantileBinning('age', 6),
        csl.datasets.utils.Binning('hours-per-week', bins = [0,40,100]),
        csl.datasets.utils.Dummify(csl.datasets.Adult.categorical + ['age', 'hours-per-week'])
        ])

    # Load Adult data
    trainset = csl.datasets.Adult(root = 'data', train = True, target_name = 'income', preprocess = preprocess,
                                  transform = csl.datasets.utils.ToTensor(dtype = torch.float),
                                  target_transform = csl.datasets.utils.ToTensor(dtype = torch.long))

    testset = csl.datasets.Adult(root = 'data', train = False, target_name = 'income', preprocess = preprocess,
                                 transform = csl.datasets.utils.ToTensor(dtype = torch.float),
                                 target_transform = csl.datasets.utils.ToTensor(dtype = torch.long))

    # Gender column index
    fullset = csl.datasets.Adult(root = 'data', train = False, target_name = 'income', preprocess = preprocess)
    gender_idx = [idx for idx, name in enumerate(fullset[0][0].columns) if name.startswith('gender')]


Defining a logistic model
^^^^^^^^^^^^^^^^^^^^^^^^^

Here we construct a simple logistic model that we will use to predict decide whether
to grant the loan by predicting if the individual makes more than US$ 50k.

.. code-block:: python
    :linenos:

    class Logistic:
        def __init__(self, n_features):
            self.parameters = [torch.zeros(1, dtype = torch.float, requires_grad = True),
                               torch.zeros([n_features,1], dtype = torch.float, requires_grad = True)]

        def __call__(self, x):
            yhat = self.logit(torch.mm(x, self.parameters[1]) + self.parameters[0])

            return torch.cat((1-yhat, yhat), dim=1)

        def predict(self, x):
            _, predicted = torch.max(self(x), 1)
            return predicted

        @staticmethod
        def logit(x):
            return 1/(1 + torch.exp(-x))


The fair classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We define the fair classification problem using the ``Logistic`` model,
the ``trainset``, and a logistic loss (see ``obj_function``). We then include
two (asymmetrical) demographic parity constraints, one for women and another for men.
The specification ``rhs`` will be passed as a parameter and ``rhs=None`` is used
to construct an unconstrained problem.

Note that since demographic parity is not differentiable
(it is the expected value of an indicator function), the constraints
use a sigmoidal approximation when ``primal`` is ``True``
(see :py:mod:`csl.problem.ConstrainedLearningProblem` for more details).

.. code-block:: python
    :linenos:

    class fairClassification(csl.ConstrainedLearningProblem):
        def __init__(self, rhs = None):
            self.model = Logistic(trainset[0][0].shape[0])
            self.data = trainset
            self.obj_function = self.loss

            if rhs is not None:
                # Gender
                self.constraints = [self.DemographicParity(self, gender_idx, 0),
                                    self.DemographicParity(self, gender_idx, 1)]
                self.rhs = [rhs, rhs]

            super().__init__()

        def loss(self, batch_idx):
            # Evaluate objective
            x, y = self.data[batch_idx]
            yhat = self.model(x)

            return F.cross_entropy(yhat, y) + 1e-3*(self.model.parameters[0]**2 + self.model.parameters[1].norm()**2)

        class DemographicParity:
            def __init__(self, problem, protected_idx, protected_value):
                self.problem = problem
                self.protected_idx = protected_idx
                self.protected_value = protected_value

            def __call__(self, batch_idx, primal):
                x, y = self.problem.data[batch_idx]

                group_idx = (x[:, self.protected_idx].squeeze() == self.protected_value)

                if primal:
                    yhat = self.problem.model(x)
                    pop_indicator = torch.sigmoid(8*(yhat[:,1] - 0.5))
                    group_indicator = torch.sigmoid(8*(yhat[group_idx,1] - 0.5))
                else:
                    yhat = self.problem.model.predict(x)
                    pop_indicator = yhat.float()
                    group_indicator = yhat[group_idx].float()

                return -(group_indicator.mean() - pop_indicator.mean())

    problems = {
       'unconstrained': fairClassification(),
      'constrained': fairClassification(rhs = 0.01),
      }


Solving the constrained learning problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now solve our constrained learning problem by constructing a primal-dual
solver and using it to solve each problem in ``problems``. Note the use of
:py:func:`csl.solver_base.PrimalDualBase.reset()` between each solve.
We save the results in ``solutions``.

.. code-block:: python
    :linenos:

    solver_settings = {'iterations': 700,
                       'batch_size': None,
                       'primal_solver': lambda p: torch.optim.Adam(p, lr=0.2),
                       'dual_solver': lambda p: torch.optim.Adam(p, lr=0.001),
                       }
    solver = csl.PrimalDual(solver_settings)

    solutions = {}
    for key, problem in problems.items():
        solver.reset()
        solver.solve(problem)
        solver.plot()

        solutions[key] = {'model': problem.model,
                         'lambdas': problem.lambdas,
                         'solver_state': solver.state_dict}



Testing the solutions
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   def accuracy(pred, y):
       correct = (pred == y).sum().item()
       return correct/pred.shape[0]

   def disparity(x, model, protected_idx, protected_value):
       pred = model.predict(x)

       pop_prev = pred.float().mean().item()

       group_idx = (fullset[:][0].iloc[:,protected_idx].squeeze() == protected_value)

       group_prev = pred[group_idx].float().mean().item()

       disparity_value = group_prev - pop_prev
       rel_disparity_value = disparity_value/pop_prev

       return disparity_value, rel_disparity_value

   for key, solution in solutions.items():
       print(f'Model: {key}')
       with torch.no_grad():
           x_test, y_test = testset[:]
           yhat = solution['model'].predict(x_test)

           acc_test = accuracy(yhat, y_test)

           disparity_f, rel_disparity_f = disparity(x_test, solution['model'], gender_idx, 0)
           disparity_m, rel_disparity_m = disparity(x_test, solution['model'], gender_idx, 1)

           print(f'Test accuracy: {100*acc_test:.2f}')
           print(f'Predicted population prevalence: {100*yhat.float().mean().item():.2f}')
           print(f'Female disparity: {100*disparity_f:.2f} | {100*rel_disparity_f:.2f}')
           print(f'Male disparity: {100*disparity_m:.2f} | {100*rel_disparity_m:.2f}')
