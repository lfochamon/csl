We will use the CIFAR-10 dataset to demonstrate how we can explicitly build
accurate models with robustness requirements. The goal is achieve the best
nominal performance possible while satisfying a constraint on the adversarial loss.

You can find more information in
`[CR, NeurIPS'20] <https://www.seas.upenn.edu/~luizf/pages/publications.html#Chamon20p>`_.

For this example, you will need to go get CIFAR-10 dataset as pytorch tensors as
described in :py:mod:`csl.datasets.datasets.CIFAR10` and it in a folder
named ``data``.

You will also need to install the `foolbox <https://foolbox.readthedocs.io/en/stable/>`_
module
::

  pip install foolbox

You can try the full code on `GitHub <https://github.com/lchamon/csl>`_.


Basic setup
^^^^^^^^^^^

.. code-block:: python
    :linenos:

    import foolbox

    import torch
    import torchvision
    import torch.nn.functional as F

    from resnet import ResNet18

    import numpy as np

    import copy

    import sys, os
    sys.path.append(os.path.abspath('../'))

    import csl, csl.datasets

    # Perturbation magnitude
    eps = 0.02

    # Use GPU if available
    theDevice = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


    ####################################
    # FUNCTIONS                        #
    ####################################
    def accuracy(yhat, y):
        _, predicted = torch.max(yhat, 1)
        correct = (predicted == y).sum().item()
        return correct/yhat.shape[0]

    def preprocess(img):
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype = img.dtype, device=theDevice).reshape((3, 1, 1))
        std = torch.tensor([0.2023, 0.1994, 0.2010], dtype = img.dtype, device=theDevice).reshape((3, 1, 1))
        return (img - mean) / std


Load data
^^^^^^^^^

We will keep a balanced 2% subset of the training data for validation.
Just to keep things realistic. We use the ``subset`` parameter to do that
(see :py:mod:`csl.datasets.datasets.CIFAR10`).

.. code-block:: python
    :linenos:

    n_train = 4900
    n_valid = 100

    target = csl.datasets.CIFAR10(root = 'data', train = True)[:][1]

    label_idx = [np.flatnonzero(target == label) for label in range(0,10)]
    label_idx = [np.random.RandomState(seed=42).permutation(idx) for idx in label_idx]
    train_subset = [idx[:n_train] for idx in label_idx]
    train_subset = np.array(train_subset).flatten()

    train_transform = torchvision.transforms.Compose([
        csl.datasets.utils.RandomFlip(),
        csl.datasets.utils.RandomCrop(size=32,padding=4),
        csl.datasets.utils.ToTensor(device=theDevice)
        ])

    trainset = csl.datasets.CIFAR10(root = 'data', train = True, subset = train_subset,
                                    transform = train_transform,
                                    target_transform = csl.datasets.utils.ToTensor(device=theDevice))

    valid_subset = [idx[n_train:n_train+n_valid] for idx in label_idx]
    valid_subset = np.array(valid_subset).flatten()
    validset = csl.datasets.CIFAR10(root = 'data', train = True, subset = valid_subset,
                                    transform = csl.datasets.utils.ToTensor(device=theDevice),
                                    target_transform = csl.datasets.utils.ToTensor(device=theDevice))



Defining the constrained learning problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two noteworthy things to be careful when encoding the constraint:

* ``foolbox`` has side-effects: it modifies the gradient of the parameters
  (even though it doesn't need to), so you need to save those gradients and
  to reload them later
* ResNets use batch normalization, which you should take into account **only**
  when optimizing the primal. So need to get the model back into train mode a
  bit earlier for the primal update.

.. code-block:: python
    :linenos:

    class robustLoss(csl.ConstrainedLearningProblem):
        def __init__(self, rhs):
            self.model = csl.PytorchModel(ResNet18().to(theDevice))
            self.data = trainset
            self.batch_size = 256

            self.obj_function = self.obj_fun

            # Constraints
            self.constraints = [self.adversarialLoss]
            self.rhs = [rhs]

            self.foolbox_model = foolbox.PyTorchModel(self.model.model, bounds=(0, 1),
                                                      device=theDevice,
                                                      preprocessing = dict(mean=[0.4914, 0.4822, 0.4465],
                                                                           std=[0.2023, 0.1994, 0.2010],
                                                                           axis=-3))
            self.attack = foolbox.attacks.LinfPGD(rel_stepsize = 1/3, abs_stepsize = None,
                                                  steps = 5, random_start = True)

            super().__init__()

        def obj_fun(self, batch_idx):
            x, y = self.data[batch_idx]

            yhat = self.model(preprocess(x))

            return 0.1*self._loss(yhat, y)

        def adversarialLoss(self, batch_idx, primal):
            x, y = self.data[batch_idx]

            # Attack
            self.model.eval()

            # Save gradients before adversarial runs
            saved_grad = [copy.deepcopy(p.grad) for p in self.model.parameters]

            # Dual is computed in a no_grad() environment
            x_processed, _, _ = self.attack(self.foolbox_model, x, y, epsilons = eps)

            # Reload gradients
            for p,g in zip(self.model.parameters, saved_grad):
                p.grad = g

            if primal:
                self.model.train()
                yhat = self.model(preprocess(x_processed))
                loss = self._loss(yhat, y)
            else:
                with torch.no_grad():
                    yhat = self.model(preprocess(x_processed))
                    loss = self._loss(yhat, y)
                self.model.train()

            return loss

        @staticmethod
        def _loss(yhat, y):
            return F.cross_entropy(yhat, y)


Setting up a validation hook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We kept some validation data to see how the model is performing on adversarial
samples during training. For that, we setup a validation hook which we can plug
as a user-defined stopping criterion (see :py:mod:`csl.solver_base.PrimalDualBase`).
We could have the solver stop depending on a value of the validation accuracy,
but here we will just let the solver do its thing and alway return ``False``.


.. code-block:: python
    :linenos:

    def validation_hook(problem, solver_state):
            adv_epoch = 10
            _adv_epoch = adv_epoch

            batch_idx = np.arange(0, len(validset)+1, problem.batch_size)
            if batch_idx[-1] < len(validset):
                batch_idx = np.append(batch_idx, len(validset))

            # Validate
            acc = 0
            acc_adv = 0
            problem.model.eval()
            for batch_start, batch_end in zip(batch_idx, batch_idx[1:]):
                x, y = validset[batch_start:batch_end]
                with torch.no_grad():
                    yhat = problem.model(preprocess(x))
                    acc += accuracy(yhat, y)*(batch_end - batch_start)/len(validset)

                # Attack
                if _adv_epoch == 1:
                    adversarial, _, _ = problem.attack(problem.foolbox_model, x, y, epsilons = max(args.eps))
                    with torch.no_grad():
                        yhat_adv = problem.model(preprocess(adversarial))
                        acc_adv += accuracy(yhat_adv, y)*(batch_end - batch_start)/len(validset)
            problem.model.train()

            # Results
            if _adv_epoch > 1:
                print(f"Validation accuracy: {acc*100:.2f} / Dual variables: {[lambda_value.item() for lambda_value in problem.lambdas]}")
                _adv_epoch -= 1
            else:
                print(f"Validation accuracy:{acc*100:.2f} / Adversarial accuracy = {acc_adv*100:.2f}")
                _adv_epoch = adv_epoch

            return False



Solving the constrained learning problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We've done most of the work above, so now we just need to call the constructors
and solve the problem.

.. code-block:: python
    :linenos:

    problem = robustLoss(rhs=0.7)

    solver_settings = {'iterations': 400,
                       'verbose': 1,
                       'batch_size': 128,
                       'primal_solver': torch.optim.Adam,
                       'lr_p0': 0.01,
                       'lr_p_scheduler': None,
                       'dual_solver': torch.optim.Adam,
                       'lr_d0': 0.001,
                       'lr_d_scheduler': None,
                       'device': theDevice,
                       'STOP_USER_DEFINED': validation_hook,
                       }
    solver = csl.SimultaneousPrimalDual(solver_settings)

    solver.solve(problem)
    solver.plot()


Testing
^^^^^^^

We can now test the results using a stronger attack than the one we used to train.

.. code-block:: python
    :linenos:

    # Test data
    testset = csl.datasets.CIFAR10(root = 'data', train = False,
                                   transform = csl.datasets.utils.ToTensor(device=theDevice),
                                   target_transform = csl.datasets.utils.ToTensor(device=theDevice))

    # Adversarial attack
    problem.model.eval()
    foolbox_model = foolbox.PyTorchModel(problem.model.model, bounds=(0, 1),
                                         device=theDevice,
                                         preprocessing = dict(mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2023, 0.1994, 0.2010],
                                                              axis=-3))
    attack = foolbox.attacks.LinfPGD(rel_stepsize = 1/30, abs_stepsize = None,
                                     steps = 50, random_start = True)
    epsilon_test = np.linspace(0.01,0.06,7)

    # Prepare batches
    batch_idx = np.arange(0, len(testset)+1, problem.batch_size)
    if batch_idx[-1] < len(testset):
        batch_idx = np.append(batch_idx, len(testset))

    n_total = 0
    acc_test = 0
    acc_adv = np.zeros(epsilon_test.shape[0])
    success_adv = np.zeros_like(acc_adv)

    for batch_start, batch_end in zip(batch_idx, batch_idx[1:]):
        x_test, y_test = testset[batch_start:batch_end]

        # Nominal accuracy
        yhat = problem.model(preprocess(x_test))
        acc_test += accuracy(yhat, y_test)*(batch_end - batch_start)

        # Adversarials accuracy
        adversarials, _, success = attack(foolbox_model, x_test, y_test, epsilons = epsilon_test)
        for ii, adv in enumerate(adversarials):
            yhat_adv = problem.model(preprocess(adv))
            acc_adv[ii] += accuracy(yhat_adv, y_test)*(batch_end - batch_start)
            success_adv[ii] += torch.sum(success[ii])

        n_total += batch_end - batch_start

    acc_test /= n_total
    acc_adv /= n_total
    success_adv /= n_total

    print('====== TEST ======')
    print(f'Test accuracy: {100*acc_test:.2f}')
    print(f'Adversarial accuracy: {100*acc_adv}')
    print(f'Adversarial success: {100*success_adv}')
